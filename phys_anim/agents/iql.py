import os.path
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from sympy.physics.units import length
from torch import nn as nn
from torch import Tensor
from torch.ao.nn.quantized.functional import clamp

from isaac_utils import torch_utils
from lightning.fabric import Fabric
from utils.StateActionLib import StateActionLib, MotionStateAction
from phys_anim.agents.models.actor import PPO_Actor
from hydra.utils import instantiate
from phys_anim.agents.models.common import NormObsBase
from phys_anim.envs.env_utils.general import StepTracker
from phys_anim.agents.models.infomax import JointDiscWithMutualInformationEncMLP
from phys_anim.envs.humanoid.humanoid_utils import build_disc_action_observations, compute_humanoid_observations_max
import math
import numpy as np
import h5py
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from utils.motion_lib import MotionLib
from isaac_utils import rotations, torch_utils
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
from poselib.core.rotation3d import quat_angle_axis, quat_inverse, quat_mul_norm

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

# jp hack
# get rid of this ASAP, need a proper way of projecting from max coords to reduced coord
def _local_rotation_to_dof(dof_body_ids,dof_offsets,num_dof, device, local_rot, joint_3d_format):
    body_ids = dof_body_ids
    dof_offsets = dof_offsets

    n = local_rot.shape[0]
    dof_pos = torch.zeros((n, num_dof), dtype=torch.float, device=device)

    for j in range(len(body_ids)):
        body_id = body_ids[j]
        joint_offset = dof_offsets[j]
        joint_size = dof_offsets[j + 1] - joint_offset

        if joint_size == 3:
            joint_q = local_rot[:, body_id]
            if joint_3d_format == "exp_map":
                formatted_joint = torch_utils.quat_to_exp_map(joint_q, w_last=True)
            elif joint_3d_format == "xyz":
                x, y, z = rotations.get_euler_xyz(joint_q, w_last=True)
                formatted_joint = torch.stack([x, y, z], dim=-1)
            else:
                raise ValueError(f"Unknown 3d format '{joint_3d_format}'")

            dof_pos[:, joint_offset: (joint_offset + joint_size)] = formatted_joint
        elif joint_size == 1:
            joint_q = local_rot[:, body_id]
            joint_theta, joint_axis = torch_utils.quat_to_angle_axis(
                joint_q, w_last=True
            )
            joint_theta = (
                    joint_theta * joint_axis[..., 1]
            )  # assume joint is always along y axis

            joint_theta = rotations.normalize_angle(joint_theta)
            dof_pos[:, joint_offset] = joint_theta

        else:
            print("Unsupported joint type")
            assert False

    return dof_pos

def _local_rotation_to_dof_vel(dof_body_ids,dof_offsets, num_dof, device, local_rot0, local_rot1, dt):
    body_ids = dof_body_ids
    dof_offsets = dof_offsets

    dof_vel = torch.zeros([num_dof], device=device)

    diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
    local_vel = local_vel

    for j in range(len(body_ids)):
        body_id = body_ids[j]
        joint_offset = dof_offsets[j]
        joint_size = dof_offsets[j + 1] - joint_offset

        if joint_size == 3:
            joint_vel = local_vel[body_id]
            dof_vel[joint_offset : (joint_offset + joint_size)] = joint_vel

        elif joint_size == 1:
            assert joint_size == 1
            joint_vel = local_vel[body_id]
            dof_vel[joint_offset] = joint_vel[
                1
            ]  # assume joint is always along y axis

        else:
            print("Unsupported joint type")
            assert False

    return dof_vel


def _compute_motion_dof_vels(dof_body_ids,dof_offsets, num_dof, device,motion: SkeletonMotion):
    num_frames = motion.global_translation.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = _local_rotation_to_dof_vel(dof_body_ids,dof_offsets, num_dof, device,local_rot0, local_rot1, dt)
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0)

    return dof_vels

class IQL:

    def __init__(self, fabric: Fabric, config):
        self.all_config = config
        self.config = config.algo.config
        self.w_last = True
        self.fabric = fabric
        self.device: torch.device = fabric.device
        self.discount = self.config.discount
        self.beta = self.config.beta

        self.num_obs = self.all_config.robot.self_obs_size
        self.num_act = self.all_config.robot.number_of_actions

        self.log_dict = {}

        self.discriminator_obs_size_per_step = (
            self.all_config.env.config.discriminator_obs_size_per_step
        )
        self.hist_obs = self.all_config.env.config.discriminator_obs_historical_steps
        self.disc_obs_size = (self.discriminator_obs_size_per_step
                               * self.all_config.env.config.discriminator_obs_historical_steps)

        self.current_epoch = 0

        self.expectile = self.config.expectile
        self.alpha = self.config.alpha

        self.setup_character_props()

        self.key_body_ids = torch.tensor(
            [
                self.all_config.robot.dfs_body_names.index(key_body_name)
                for key_body_name in self.all_config.robot.key_bodies
            ],
            dtype=torch.long,
        )

        self.demo_dataset_files = []
        for path in self.all_config.algo.config.demo_dataset_files:
            self.demo_dataset_files.append(h5py.File(path, "r"))

        self.dataset_files = []
        for path in self.all_config.algo.config.dataset_files:
            self.dataset_files.append(h5py.File(path, "r"))

        self.dataset_len = self.dataset_files[0]["dones"].shape[0]

        self.skeleton_tree = SkeletonTree.from_mjcf('phys_anim/data/assets/mjcf/amp_humanoid_sword_shield.xml')

        self.demo_dataset = {
            "disc_obs": torch.zeros(self.dataset_len, self.disc_obs_size, device=self.device),
            "actions": torch.zeros(self.dataset_len, self.num_act, device=self.device),
        }
        self.dataset = {}

        self.update_steps_per_stage = 1

        pass

    def fill_dataset(self):
        filled = 0
        demo_disc_obs = []
        demo_actions = []
        while filled < self.dataset_len:
            file_rand = random.choice(self.demo_dataset_files)
            motion_end = min(self.dataset_len - filled, file_rand['actions'].shape[0])
            actions = torch.from_numpy(file_rand['actions'][0:motion_end,0,...]).to(self.device)
            disc_obs = torch.from_numpy(file_rand['disc_obs'][0:motion_end,0,...]).to(self.device)
            demo_disc_obs.append(self.make_with_hist_obs(disc_obs))
            demo_actions.append(actions)
            filled += motion_end

        self.demo_dataset['disc_obs'] = torch.cat(demo_disc_obs, dim=0)
        self.demo_dataset['actions'] = torch.cat(demo_actions, dim=0)

        file_rand = random.choice(self.dataset_files)
        env_rand = random.randint(0,file_rand['global_rot'].shape[1] - 1)
        global_rot = torch.from_numpy(file_rand['global_rot'][:, env_rand, ...])
        root_pos = torch.from_numpy(file_rand['root_pos'][:, env_rand, ...])
        sk_state = SkeletonState.from_rotation_and_root_translation(
            self.skeleton_tree,
            global_rot,
            root_pos,
            is_local=False
        )
        sk_motion = SkeletonMotion.from_skeleton_state(sk_state, 30)
        root_pos = sk_motion.root_translation.to(self.device)
        root_rot = sk_motion.global_root_rotation.to(self.device)
        root_vel = sk_motion.global_root_velocity.to(self.device)
        root_ang_vel = sk_motion.global_root_angular_velocity.to(self.device)
        dof_pos = _local_rotation_to_dof(dof_body_ids=self.dof_body_ids,
                                         dof_offsets=self.dof_offsets,
                                         num_dof=self.num_act,
                                         device=self.device,
                                         local_rot=sk_motion.local_rotation,
                                         joint_3d_format='exp_map', ).to(self.device)
        dof_vel = torch.from_numpy(file_rand['dof_vel'][:, env_rand, ...]).to(self.device)
        key_body_pos = sk_motion.global_translation[:, self.key_body_ids].to(self.device)
        actions = torch.from_numpy(file_rand['actions'][:, env_rand, ...]).to(self.device)
        self.dataset['root_pos'] = self.make_with_hist_obs(root_pos, flatten=False)
        self.dataset['root_rot'] = self.make_with_hist_obs(root_rot, flatten=False)
        self.dataset['root_vel'] = self.make_with_hist_obs(root_vel, flatten=False)
        self.dataset['root_ang_vel'] = self.make_with_hist_obs(root_ang_vel, flatten=False)
        self.dataset['dof_pos'] = self.make_with_hist_obs(dof_pos, flatten=False)
        self.dataset['dof_vel'] = self.make_with_hist_obs(dof_vel, flatten=False)
        self.dataset['key_body_pos'] = self.make_with_hist_obs(key_body_pos, flatten=False)
        disc_obs = build_disc_action_observations(
                                            root_pos,
                                            root_rot,
                                            root_vel,
                                            root_ang_vel,
                                            dof_pos,
                                            dof_vel,
                                            key_body_pos,
                                            torch.zeros(1, device=self.device),
                                            actions,
                                            self.all_config.env.config.humanoid_obs.local_root_obs,
                                            self.all_config.env.config.humanoid_obs.root_height_obs,
                                            self.all_config.robot.dof_obs_size,
                                            self.dof_offsets,
                                            False,
                                            self.w_last,
                                        )
        self.dataset['disc_obs'] = self.make_with_hist_obs(disc_obs)
        human_obs = compute_humanoid_observations_max(
                                            sk_motion.global_translation.to(self.device),
                                            sk_motion.global_rotation.to(self.device),
                                            sk_motion.global_velocity.to(self.device),
                                            sk_motion.global_angular_velocity.to(self.device),
                                            torch.zeros(1, device=self.device),
                                            self.all_config.env.config.humanoid_obs.local_root_obs,
                                            self.all_config.env.config.humanoid_obs.root_height_obs,
                                            self.w_last,
                                        )
        self.dataset['human_obs'] = self.make_with_hist_obs(human_obs, flatten=False)
        self.dataset['actions'] = actions
        self.dataset['dones'] = torch.from_numpy(file_rand['dones'][:, env_rand, ...]).to(self.device)
        self.dataset['next_human_obs'] = torch.roll(human_obs, shifts=-1, dims=0)


    def setup(self):
        actor: PPO_Actor = instantiate(
            self.config.actor, num_in=self.num_obs, num_act=self.num_act
        )

        actor_optimizer = instantiate(
            self.config.actor_optimizer,
            params=list(actor.parameters()),
            _convert_="all",
        )

        self.actor, self.actor_optimizer = self.fabric.setup(actor, actor_optimizer)
        self.actor.mark_forward_method("eval_forward")
        self.actor.mark_forward_method("training_forward")

        qf1: NormObsBase = instantiate(
            self.config.critic_sa, num_in=self.num_obs, num_out=1
        )
        qf1_optimizer = instantiate(
            self.config.critic_optimizer,
            params=list(qf1.parameters()),
        )
        self.qf1, self.qf1_optimizer = self.fabric.setup(qf1, qf1_optimizer)

        qf2: NormObsBase = instantiate(
            self.config.critic_sa, num_in=self.num_obs, num_out=1
        )
        qf2_optimizer = instantiate(
            self.config.critic_optimizer,
            params=list(qf2.parameters()),
        )
        self.qf2, self.qf2_optimizer = self.fabric.setup(qf2, qf2_optimizer)

        target_qf1: NormObsBase = instantiate(
            self.config.critic_sa, num_in=self.num_obs, num_out=1
        )
        target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf1 = self.fabric.setup(target_qf1)

        target_qf2: NormObsBase = instantiate(
            self.config.critic_sa, num_in=self.num_obs, num_out=1
        )
        target_qf2.load_state_dict(self.qf2.state_dict())
        self.target_qf2 = self.fabric.setup(target_qf2)

        vf: NormObsBase = instantiate(
            self.config.critic_s, num_in=self.num_obs, num_out=1
        )
        vf_optimizer = instantiate(
            self.config.critic_optimizer,
            params=list(vf.parameters()),
        )
        self.vf, self.vf_optimizer = self.fabric.setup(vf, vf_optimizer)

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        discriminator: JointDiscWithMutualInformationEncMLP = instantiate(
            self.config.discriminator,
            num_in=self.discriminator_obs_size_per_step
                   * self.all_config.env.config.discriminator_obs_historical_steps,
        )
        discriminator_optimizer = instantiate(
            self.config.discriminator_optimizer,
            params=discriminator.parameters(),
        )

        self.discriminator, self.discriminator_optimizer = self.fabric.setup(
            discriminator, discriminator_optimizer
        )

        self._n_train_steps_total = 0
        self.q_update_period = 1
        self.policy_update_period = 1
        self.target_update_period = 1
        self.disc_update_period = 1

        #state_dict = torch.load(Path.cwd() / "results/iql/lightning_logs/version_1/last.ckpt", map_location=self.device)
        #self.actor.load_state_dict(state_dict["actor"])
        #self.save(name="last_a.ckpt")

    def fit(self):

        for self.current_epoch in range(self.config.max_epochs):
            print(f"Epoch: {self.current_epoch}")
            batch_count = math.ceil(self.dataset_len / self.config.batch_size)

            print('Fill Dataset')
            self.fill_dataset()
            self.dataset["latents"] = self.sample_latent(self.dataset_len)

            v_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            q_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            desciptor_r = torch.zeros(self.update_steps_per_stage * batch_count)
            enc_r = torch.zeros(self.update_steps_per_stage * batch_count)
            total_r = torch.zeros(self.update_steps_per_stage * batch_count)

            a_loss_tensor_adw_exp = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw_neglog = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw_b_c = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_div = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_total = torch.zeros(self.update_steps_per_stage * batch_count)

            for i in range(self.update_steps_per_stage):
                for batch_id in range(batch_count):
                    print(f'Batch: {batch_id}')

                    indices = torch.randperm(len(self.dataset['human_obs']))[:self.config.batch_size]

                    batch = {
                        "latents": self.dataset["latents"][indices],
                        "root_pos": self.dataset["root_pos"][indices],
                        "root_rot": self.dataset["root_rot"][indices],
                        "root_vel": self.dataset["root_vel"][indices],
                        "root_ang_vel": self.dataset["root_ang_vel"][indices],
                        "dof_pos": self.dataset["dof_pos"][indices],
                        "dof_vel": self.dataset["dof_vel"][indices],
                        "dof_vel": self.dataset["dof_vel"][indices],
                        "key_body_pos": self.dataset["key_body_pos"][indices],
                        "disc_obs": self.dataset["disc_obs"][indices],
                        "human_obs": self.dataset["human_obs"][indices],
                        "actions": self.dataset["actions"][indices],
                        'next_human_obs': self.dataset["next_human_obs"][indices],
                        'dones': self.dataset['dones'][indices]
                    }

                    demo_indices = torch.randperm(len(self.demo_dataset['disc_obs']))[:self.config.batch_size]

                    demo_batch = {
                        "disc_obs": self.demo_dataset["disc_obs"][demo_indices],
                        "actions": self.demo_dataset["actions"][demo_indices]
                    }

                    next_obs = batch["next_human_obs"]
                    next_latents = torch.roll(batch["latents"], shifts=-1, dims=0)


                    desc_r = self.calculate_discriminator_reward(
                        batch["disc_obs"]).squeeze()  # torch.ones(self.config.batch_size, device=self.device)
                    mi_r = self.calc_mi_reward(batch["disc_obs"], batch["latents"])

                    reward = desc_r.detach() + mi_r.detach() + 1

                    """
                    QF Loss
                    """
                    print("Q step")
                    q1_pred = self.qf1({"obs": batch["human_obs"][:, 0], "actions": batch["actions"],
                         "latents": batch["latents"]})
                    q2_pred = self.qf2({"obs": batch["human_obs"][:, 0], "actions": batch["actions"],
                         "latents": batch["latents"]})
                    target_vf_pred = self.vf({"obs": next_obs, "latents": next_latents}).detach()

                    q_target = reward + (1. - batch['dones']) * self.discount * target_vf_pred
                    q_target = q_target.detach()
                    qf1_loss = self.qf_criterion(q1_pred, q_target)
                    qf2_loss = self.qf_criterion(q2_pred, q_target)

                    """
                    VF Loss
                    """
                    print("V step")
                    q_pred = torch.min(
                        self.target_qf1({"obs": batch["human_obs"][:, 0], "actions": batch["actions"],
                         "latents": batch["latents"]}),
                        self.target_qf2({"obs": batch["human_obs"][:, 0], "actions": batch["actions"],
                         "latents": batch["latents"]}),
                    ).detach()
                    vf_pred = self.vf({"obs": batch["human_obs"][:, 0], "latents": batch["latents"]})
                    vf_err = vf_pred - q_pred
                    vf_sign = (vf_err > 0).float()
                    vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
                    vf_loss = (vf_weight * (vf_err ** 2)).mean()

                    """
                    Policy Loss
                    """
                    print("P step")
                    self.actor.training = True
                    actor_out = self.actor.training_forward(
                        {"obs": batch["human_obs"][:,0],
                         "actions": batch["actions"],
                         "latents": batch["latents"]})

                    policy_logpp = -actor_out["neglogp"]

                    adv = q_pred - vf_pred
                    exp_adv = torch.exp(adv / self.beta)
                    exp_adv = torch.clamp(exp_adv, max=100)

                    weights = exp_adv.detach()  # exp_adv[:, 0].detach()
                    actor_adw_loss = (-policy_logpp * weights).mean()

                    actor_div_loss, div_loss_log = self.calculate_extra_actor_loss(
                        {"obs": batch["human_obs"][:,0], "latents": batch["latents"],
                         "actions": batch["actions"]})

                    actor_loss = actor_adw_loss + actor_div_loss

                    """
                    Disc Step
                    """
                    print("D step")
                    latents = self.make_with_hist_obs(batch["latents"] ,flatten=False).reshape(self.config.batch_size * self.hist_obs, -1)

                    self.actor.training = False
                    with torch.no_grad():
                        actor_eval_out = self.actor.eval_forward(
                            {"obs": batch["human_obs"].view(self.config.batch_size * self.hist_obs, -1), "latents": latents})

                    root_pos = batch["root_pos"].view(self.config.batch_size * self.hist_obs, -1)
                    root_rot = batch["root_rot"].view(self.config.batch_size * self.hist_obs, -1)
                    root_vel = batch["root_vel"].view(self.config.batch_size * self.hist_obs, -1)
                    root_ang_vel = batch["root_ang_vel"].view(self.config.batch_size * self.hist_obs, -1)
                    dof_pos = batch["dof_pos"].view(self.config.batch_size * self.hist_obs, -1)
                    dof_vel = batch["dof_vel"].view(self.config.batch_size * self.hist_obs, -1)
                    key_body_pos = batch["key_body_pos"].view(self.config.batch_size * self.hist_obs, 2, 3)

                    agent_disc_obs = build_disc_action_observations(
                        root_pos,
                        root_rot,
                        root_vel,
                        root_ang_vel,
                        dof_pos,
                        dof_vel,
                        key_body_pos,
                        torch.zeros(1, device=self.device),
                        actor_eval_out["actions"],
                        self.all_config.env.config.humanoid_obs.local_root_obs,
                        self.all_config.env.config.humanoid_obs.root_height_obs,
                        self.all_config.robot.dof_obs_size,
                        self.dof_offsets,
                        False,
                        self.w_last,
                    )

                    agent_disc_obs = agent_disc_obs.view(self.config.batch_size, self.hist_obs, -1)
                    agent_disc_obs = agent_disc_obs.view(self.config.batch_size, -1)

                    disc_loss, disc_log_dict = self.encoder_step(
                        {"AgentDiscObs": agent_disc_obs, "DemoDiscObs": demo_batch["disc_obs"],
                         "latents": batch["latents"]})

                    """
                    Update networks
                    """
                    if self._n_train_steps_total % self.q_update_period == 0:
                        self.qf1_optimizer.zero_grad()
                        qf1_loss.backward()
                        self.qf1_optimizer.step()

                        self.qf2_optimizer.zero_grad()
                        qf2_loss.backward()
                        self.qf2_optimizer.step()

                        self.vf_optimizer.zero_grad()
                        vf_loss.backward()
                        self.vf_optimizer.step()

                    if self._n_train_steps_total % self.policy_update_period == 0:
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                    if self._n_train_steps_total % self.disc_update_period == 0:
                        self.discriminator_optimizer.zero_grad()
                        self.fabric.backward(disc_loss)
                        self.discriminator_optimizer.step()

                    """
                    Soft Updates
                    """
                    if self._n_train_steps_total % self.target_update_period == 0:
                        soft_update_from_to(
                            self.qf1, self.target_qf1, self.alpha
                        )
                        soft_update_from_to(
                            self.qf2, self.target_qf2, self.alpha
                        )

                    self._n_train_steps_total += 1

                    """
                    Log
                    """
                    desciptor_r[batch_id * self.update_steps_per_stage + i] = desc_r.mean().detach()
                    enc_r[batch_id * self.update_steps_per_stage + i] = mi_r.mean().detach()
                    total_r[batch_id * self.update_steps_per_stage + i] = reward.mean().detach()

                    q_loss_tensor[batch_id * self.update_steps_per_stage + i] = (qf1_loss.mean().detach() + qf2_loss.mean().detach()) / 2.

                    v_loss_tensor[batch_id * self.update_steps_per_stage + i] = vf_loss.mean().detach()

                    a_loss_tensor_adw[batch_id * self.update_steps_per_stage + i] = actor_adw_loss.mean().detach()
                    a_loss_tensor_div[batch_id * self.update_steps_per_stage + i] = actor_div_loss.mean().detach()
                    a_loss_tensor_total[batch_id * self.update_steps_per_stage + i] = actor_loss.mean().detach()
                    a_loss_tensor_adw_exp[batch_id * self.update_steps_per_stage + i] = exp_adv.mean().detach()
                    a_loss_tensor_adw_neglog[batch_id * self.update_steps_per_stage + i] = actor_out[
                        "neglogp"].mean().detach()

            self.log_dict.update({
                "reward/desc": desciptor_r.mean(),
                "reward/end": enc_r.mean(),
                "reward/total": total_r.mean(),
                "ac/v_loss": v_loss_tensor.mean(),
                "ac/q_loss": q_loss_tensor.mean(),
                "actor_loss/adw_exp": a_loss_tensor_adw_exp.mean(),
                "actor_loss/neg_log": a_loss_tensor_adw_neglog.mean(),
                "actor_loss/adw_before_clip": a_loss_tensor_adw_b_c.mean(),
                "actor_loss/adw_after_clip": a_loss_tensor_adw.mean(),
                "actor_loss/div": a_loss_tensor_div.mean(),
                "actor_loss/total": a_loss_tensor_total.mean(),
            })
            self.log_dict.update(disc_log_dict)

            self.fabric.log_dict(self.log_dict, self.current_epoch)

            if self.current_epoch % 10 == 0:
                self.save()

    def sample_latent(self, n):
        latents = torch.zeros(
            [n, sum(self.config.infomax_parameters.latent_dim)], device=self.device
        )

        cur_ind = 0
        while cur_ind < latents.shape[0]:
            len = random.randint(1, 150)
            end = np.clip(cur_ind + len, 0, latents.shape[0])
            gaussian_sample = torch.normal(latents[:, cur_ind: end])
            projected_gaussian_sample = torch.nn.functional.normalize(
                gaussian_sample, dim=-1
            )
            latents[:, cur_ind: end] = projected_gaussian_sample
            cur_ind += len

        '''
        start = 0
        for idx, dim in enumerate(self.config.infomax_parameters.latent_dim):
            if self.config.infomax_parameters.latent_types[idx] == "gaussian":
                gaussian_sample = torch.normal(latents[:, start: start + dim])
                latents[:, start: start + dim] = gaussian_sample

            elif self.config.infomax_parameters.latent_types[idx] == "hypersphere":
                gaussian_sample = torch.normal(latents[:, start: start + dim])
                projected_gaussian_sample = torch.nn.functional.normalize(
                    gaussian_sample, dim=-1
                )
                latents[:, start: start + dim] = projected_gaussian_sample

            elif self.config.infomax_parameters.latent_types[idx] == "uniform":
                uniform_sample = torch.rand([n, dim], device=self.device)
                latents[:, start: start + dim] = uniform_sample

            elif self.config.infomax_parameters.latent_types[idx] == "categorical":
                categorical_sample = torch.multinomial(
                    latents[0, start: start + dim] + 1.0,
                    num_samples=n,
                    replacement=True,
                )
                b = torch.arange(n, device=self.device)
                latents[b, categorical_sample + start] = categorical_sample
            else:
                raise NotImplementedError

            start += dim'''

        return latents

    # aka calculate diversity_loss
    def calculate_extra_actor_loss(self, batch_dict, eval=False) -> Tuple[Tensor, Dict]:
        extra_loss, extra_actor_log_dict = torch.tensor(0.0, device=self.device), {}

        if self.config.infomax_parameters.diversity_bonus <= 0:
            return extra_loss, extra_actor_log_dict

        diversity_loss = self.diversity_loss(batch_dict, eval)

        extra_actor_log_dict["actor/diversity_loss"] = diversity_loss.detach()

        return (
            extra_loss
            + diversity_loss * self.config.infomax_parameters.diversity_bonus,
            extra_actor_log_dict,
        )

    def diversity_loss(self, batch_dict, eval=False):
        prev_latents = batch_dict["latents"]
        new_latents = (self.sample_latent(batch_dict["obs"].shape[0]))
        batch_dict["latents"] = new_latents
        if not eval:
            new_outs = self.actor.training_forward(batch_dict)
        else:
            new_outs = self.actor.eval_forward(batch_dict)

        batch_dict["latents"] = prev_latents
        if not eval:
            old_outs = self.actor.training_forward(batch_dict)
        else:
            old_outs = self.actor.eval_forward(batch_dict)

        clipped_new_mu = torch.clamp(new_outs["mus"], -1.0, 1.0)
        clipped_old_mu = torch.clamp(old_outs["mus"], -1.0, 1.0)

        mu_diff = clipped_new_mu - clipped_old_mu
        mu_diff = torch.mean(torch.square(mu_diff), dim=-1)

        z_diff = new_latents * prev_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = mu_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(
            self.config.infomax_parameters.diversity_tar - diversity_bonus
        ).mean()

        return diversity_loss

    def mi_enc_forward(self, obs: Tensor) -> Tensor:
        args = {"obs": obs}
        return self.discriminator(args, return_enc=True)

    def calc_mi_reward(self, discriminator_obs, latents):
        """
        TODO: calculate reward for each distribution type
            Gaussian -- MSE (we assume variance 1)
            Hypersphere -- von Mises-Fisher
            Uniform -- MSE
            Categorical -- torch.nn.functional.cross_entropy()
        """
        mi_r = torch.zeros(self.config.batch_size, device=self.device)

        enc_pred = self.mi_enc_forward(discriminator_obs)
        cumulative_enc_dim = 0
        cumulative_latent_dim = 0
        for idx, latent_dim in enumerate(self.config.infomax_parameters.latent_dim):
            if self.config.infomax_parameters.latent_types[idx] == "hypersphere":
                r = self.von_mises_fisher_reward(
                    enc_pred[..., cumulative_enc_dim: cumulative_enc_dim + latent_dim],
                    latents[
                    ..., cumulative_latent_dim: cumulative_latent_dim + latent_dim
                    ],
                )

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim
            else:
                raise NotImplementedError

            r = r.squeeze(1)

            mi_r += r * self.config.infomax_parameters.mi_reward_w[idx]

        return mi_r / len(self.config.infomax_parameters.latent_dim)

    def von_mises_fisher_reward(self, enc_prediction, latents):
        neg_err = -self.calc_von_mises_fisher_enc_error(enc_prediction, latents)
        if self.config.infomax_parameters.mi_hypersphere_reward_shift:
            mi_r = (neg_err + 1) / 2
        else:
            mi_r = torch.clamp_min(neg_err, 0.0)
        return mi_r

    def calc_von_mises_fisher_enc_error(self, enc_pred, latent):
        err = enc_pred * latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def setup_character_props(self):
        self.dof_body_ids = self.all_config.robot.dfs_dof_body_ids
        self.dof_offsets = []
        previous_dof_name = "null"
        for dof_offset, dof_name in enumerate(self.all_config.robot.dfs_dof_names):
            if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
                previous_dof_name = dof_name[:-2]
                self.dof_offsets.append(dof_offset)
        self.dof_offsets.append(len(self.all_config.robot.dfs_dof_names))
        self.dof_obs_size = self.all_config.robot.dof_obs_size
        self.num_act = self.all_config.robot.number_of_actions

    def build_body_ids_tensor(self, body_names):
        body_ids = []

        for body_name in body_names:
            body_id = self.body_names.index(body_name)
            assert (
                    body_id != -1
            ), f"Body part {body_name} not found in {self.body_names}"
            body_ids.append(body_id)

        body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def discriminator_forward(self, obs: Tensor, return_norm_obs=False) -> Tensor:
        args = {"obs": obs}
        return self.discriminator(args, return_norm_obs=return_norm_obs)

    def calculate_discriminator_reward(self, discriminator_obs: Tensor) -> Tensor:
        disc_logits = self.discriminator_forward(discriminator_obs)

        prob = 1 / (1 + torch.exp(-disc_logits))
        disc_r = (
                -torch.log(
                    torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device))
                )
                * self.config.discriminator_reward_w
        )
        return disc_r

    # batch:{
    #   "AgentDiscObs"
    #   "DemoDiscObs"
    # }
    def discriminator_step(self, batch):
        discriminator_loss, discriminator_log_dict = self.compute_discriminator_loss(
            batch
        )

        discriminator_log_dict = {
            f"jd/{k}": v for k, v in discriminator_log_dict.items()
        }

        return discriminator_loss, discriminator_log_dict

    def compute_discriminator_loss(self, batch):
        (
            agent_obs,
            demo_obs,
        ) = batch["AgentDiscObs"], batch["DemoDiscObs"]

        demo_obs.requires_grad_(True)

        agent_logits = self.discriminator_forward(obs=agent_obs)

        demo_dict = self.discriminator_forward(obs=demo_obs, return_norm_obs=True)
        demo_logits = demo_dict["outs"]
        demo_norm_obs = demo_dict["norm_obs"]

        pos_loss = self.disc_loss_pos(demo_logits)
        agent_loss = self.disc_loss_neg(agent_logits)

        neg_loss = agent_loss

        class_loss = 0.5 * (pos_loss + neg_loss)

        pos_acc = self.compute_pos_acc(demo_logits)
        agent_acc = self.compute_neg_acc(agent_logits)

        neg_acc = agent_acc

        # grad penalty
        disc_demo_grad = torch.autograd.grad(
            demo_logits,
            demo_norm_obs,
            grad_outputs=torch.ones_like(demo_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        disc_demo_grad_norm = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad_norm)
        grad_loss: Tensor = self.config.discriminator_grad_penalty * disc_grad_penalty

        if self.config.discriminator_weight_decay > 0:
            all_weight_params = self.discriminator.all_discriminator_weights()
            total: Tensor = sum([p.pow(2).sum() for p in all_weight_params])
            weight_decay_loss: Tensor = total * self.config.discriminator_weight_decay
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)
            total = torch.tensor(0.0, device=self.device)

        if self.config.discriminator_logit_weight_decay > 0:
            logit_params = self.discriminator.logit_weights()
            logit_total = sum([p.pow(2).sum() for p in logit_params])

            logit_weight_decay_loss: Tensor = (
                    logit_total * self.config.discriminator_logit_weight_decay
            )
        else:
            logit_weight_decay_loss = torch.tensor(0.0, device=self.device)
            logit_total = torch.tensor(0.0, device=self.device)

        loss = grad_loss + class_loss + weight_decay_loss + logit_weight_decay_loss

        log_dict = {
            "loss": loss.detach(),
            "pos_acc": pos_acc.detach(),
            "agent_acc": agent_acc.detach(),
            "neg_acc": neg_acc.detach(),
            "grad_penalty": disc_grad_penalty.detach(),
            "grad_loss": grad_loss.detach(),
            "class_loss": class_loss.detach(),
            "l2_logit_total": logit_total.detach(),
            "l2_logit_loss": logit_weight_decay_loss.detach(),
            "l2_total": total.detach(),
            "l2_loss": weight_decay_loss.detach(),
            "demo_logit_mean": demo_logits.detach().mean(),
            "agent_logit_mean": agent_logits.detach().mean(),
        }

        log_dict["negative_logit_mean"] = log_dict["agent_logit_mean"]

        return loss, log_dict

    # batch:{
    #   "AgentDiscObs"
    #   "DemoDiscObs"
    #   "latents"
    # }
    def encoder_step(self, batch) -> Tuple[Tensor, Dict]:
        discriminator_loss, discriminator_log_dict = self.discriminator_step(batch)

        obs = batch["AgentDiscObs"]
        latents = batch["latents"]
        if self.config.infomax_parameters.mi_enc_grad_penalty > 0:
            obs.requires_grad_(True)

        mi_enc_pred = self.mi_enc_forward(obs)

        mi_enc_err = self.calc_mi_enc_error(mi_enc_pred, latents)

        mi_enc_loss = torch.mean(mi_enc_err)

        if self.config.infomax_parameters.mi_enc_weight_decay > 0:
            enc_weight_params = self.discriminator.enc_weights()
            total: Tensor = sum([p.pow(2).sum() for p in enc_weight_params])
            weight_decay_loss: Tensor = (
                    total * self.config.infomax_parameters.mi_enc_weight_decay
            )
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)

        if self.config.infomax_parameters.mi_enc_grad_penalty > 0:
            mi_enc_obs_grad = torch.autograd.grad(
                mi_enc_err,
                obs,
                grad_outputs=torch.ones_like(mi_enc_err),
                create_graph=True,
                retain_graph=True,
            )

            mi_enc_obs_grad = mi_enc_obs_grad[0]
            mi_enc_obs_grad = torch.sum(torch.square(mi_enc_obs_grad), dim=-1)
            mi_enc_grad_penalty = torch.mean(mi_enc_obs_grad)

            grad_loss: Tensor = mi_enc_grad_penalty * self.config.infomax_parameters.mi_enc_grad_penalty
        else:
            grad_loss = torch.tensor(0.0, device=self.device)

        mi_loss = mi_enc_loss + weight_decay_loss + grad_loss

        log_dict = {
            "loss": mi_loss.detach(),
            "l2_loss": weight_decay_loss.detach(),
            "grad_penalty": grad_loss.detach(),
        }

        mi_enc_log_dict = {f"mi_enc/{k}": v for k, v in log_dict.items()}

        discriminator_log_dict.update(mi_enc_log_dict)

        return mi_loss + discriminator_loss, discriminator_log_dict

    def calc_mi_enc_error(self, enc_pred, latents):
        cumulative_enc_dim = 0
        cumulative_latent_dim = 0

        total_error = []

        for idx, latent_dim in enumerate(self.config.infomax_parameters.latent_dim):
            if self.config.infomax_parameters.latent_types[idx] == "hypersphere":
                err = self.calc_von_mises_fisher_enc_error(enc_pred, latents)

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim
            else:
                raise NotImplementedError

            total_error.append(err)

        return torch.cat(total_error, dim=-1)

    def make_with_hist_obs(self, obs: torch.Tensor, flatten = True):
        """
        Constructs a tensor containing historical observations for each time step.

        Args:
            obs (torch.Tensor): Tensor of shape (N_steps, ...).
            reset (torch.Tensor, optional): Unused in current logic.

        Returns:
            torch.Tensor: Tensor of shape (N_steps, hist_steps, ...) with historical context,
                          or (N_steps, hist_steps * prod(obs.shape[1:])) if flattened.
        """
        N_steps = obs.shape[0]
        hist_steps = self.all_config.env.config.discriminator_obs_historical_steps
        obs_shape = obs.shape[1:]

        # Padding with zeros in time dimension
        padding = torch.zeros((hist_steps - 1, *obs_shape), device=obs.device, dtype=obs.dtype)
        padded_obs = torch.cat([padding, obs], dim=0)  # Shape: (N_steps + hist_steps - 1, ...)

        # Create indices to extract historical windows
        indices = (torch.arange(N_steps, device=obs.device).unsqueeze(1) +
                   torch.arange(hist_steps, device=obs.device).flip(0))  # Shape: (N_steps, hist_steps)

        # Gather the slices
        temp = padded_obs[indices]  # Shape: (N_steps, hist_steps, ...)

        # Flatten the last two dimensions if needed
        return temp.reshape(N_steps, -1) if flatten else temp  # Or return `temp` directly if not flattening

    def get_state_dict(self, state_dict):
        extra_state_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.target_qf1.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.qf1_optimizer.state_dict(),
            "epoch": self.current_epoch,
            "episode_reward_meter": None,
            "episode_length_meter": None,
            "best_evaluated_score": 0,
        }

        state_dict["running_val_norm"] = None

        state_dict["discriminator"] = self.discriminator.state_dict()
        state_dict["discriminator_optimizer"] = (
            self.discriminator_optimizer.state_dict()
        )

        state_dict.update(extra_state_dict)
        return state_dict

    def save(self, path=None, name="last.ckpt", new_high_score=False):
        if path is None:
            path = self.fabric.loggers[0].log_dir
        root_dir = Path.cwd() / Path(self.fabric.loggers[0].root_dir)
        save_dir = Path.cwd() / Path(path)
        state_dict = self.get_state_dict({})
        self.fabric.save(save_dir / name, state_dict)

    @staticmethod
    def disc_loss_neg(disc_logits) -> Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            disc_logits, torch.zeros_like(disc_logits)
        )
        return loss

    @staticmethod
    def disc_loss_pos(disc_logits) -> Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            disc_logits, torch.ones_like(disc_logits)
        )
        return loss

    @staticmethod
    def compute_pos_acc(positive_logit: Tensor) -> Tensor:
        return (positive_logit > 0).float().mean()

    @staticmethod
    def compute_neg_acc(negative_logit: Tensor) -> Tensor:
        return (negative_logit < 0).float().mean()
