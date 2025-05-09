import os.path
import random
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch import nn as nn
from torch import Tensor
from torch.cuda import device

import torch.nn.functional as F
from isaac_utils import torch_utils
from lightning.fabric import Fabric
from utils.motion_lib import MotionLib
from utils.StateActionLib import StateActionLib, MotionStateAction
from phys_anim.agents.models.actor import PPO_Actor
from hydra.utils import instantiate
from phys_anim.agents.models.common import NormObsBase
from phys_anim.envs.env_utils.general import StepTracker
from phys_anim.agents.models.infomax import JointDiscWithMutualInformationEncMLP
from phys_anim.agents.models.mlp import MultiHeadedMLP, MLP_WithNorm
from phys_anim.agents.models.discriminator import JointDiscMLP
from phys_anim.envs.humanoid.humanoid_utils import build_disc_observations, build_disc_action_observations, \
    compute_humanoid_observations_max
import math
import numpy as np

import h5py
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from utils.motion_lib import MotionLib
from isaac_utils import rotations, torch_utils
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
from poselib.core.rotation3d import quat_angle_axis, quat_inverse, quat_mul_norm

from .iql import _local_rotation_to_dof, soft_update_from_to

from tqdm import tqdm

# def list_roll(inlist, n):
#    for i in range(n):
#        inlist.append(inlist.pop(0))

class IQL_Fenc:

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

        self.discriminator_obs_historical_steps = self.all_config.env.config.discriminator_obs_historical_steps

        self.current_epoch = 0

        self.expectile = self.config.expectile
        self.alpha = self.config.alpha

        self.latent_reset_steps = StepTracker(
            1,
            min_steps=self.config.infomax_parameters.latent_steps_min,
            max_steps=self.config.infomax_parameters.latent_steps_max,
            device=self.device,
        )

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

        self.skeleton_tree = SkeletonTree.from_mjcf('phys_anim/data/assets/mjcf/amp_humanoid_sword_shield.xml')

        self.files_envs_batches = []
        self.batch_count = 0

        pass

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

        discriminator: JointDiscMLP = instantiate(
            self.config.discriminator,
            num_in=self.discriminator_obs_size_per_step
                   * self.discriminator_obs_historical_steps,
            num_out=1
        )
        discriminator_optimizer = instantiate(
            self.config.discriminator_optimizer,
            params=list(discriminator.parameters()),
        )

        self.discriminator, self.discriminator_optimizer = self.fabric.setup(
            discriminator, discriminator_optimizer
        )

        encoder: MLP_WithNorm = instantiate(
            self.config.encoder,
            num_in=self.discriminator_obs_size_per_step
                   * 2,
            num_out=2 * sum(self.config.infomax_parameters.latent_dim)
        )

        encoder_optimizer = instantiate(
            self.config.discriminator_optimizer,
            params=list(encoder.parameters()),
        )

        self.encoder, self.encoder_optimizer = self.fabric.setup(
            encoder, encoder_optimizer
        )

        decoder: MLP_WithNorm = instantiate(
            self.config.decoder,
            num_in=sum(self.config.infomax_parameters.latent_dim),
            num_out=self.discriminator_obs_size_per_step
                   * 2
        )

        decoder_optimizer = instantiate(
            self.config.discriminator_optimizer,
            params=list(decoder.parameters()),
        )

        self.decoder, self.decoder_optimizer = self.fabric.setup(
            decoder, decoder_optimizer
        )

        self.enc_loss_fn = nn.MSELoss()

        self.q_update_period = 1
        self.policy_update_period = 1
        self.target_update_period = 1

        #state_dict = torch.load(Path.cwd() / "results/iql_calm/lightning_logs/version_5/last.ckpt", map_location=self.device)
        #self.actor.load_state_dict(state_dict["actor"])
        #self.save(name="last_a.ckpt")

    def start_epoch(self):
        """
            dataset_envs
            {
                0, file0:
                {
                    0, env0:
                    {
                        batch0:{...}
                        batch1:{...}
                    },
                    1, env1:
                    {
                        batch0:{...}
                        batch1:{...}
                    }
                },
                1, file1:
                {
                    0, env0:
                    {
                        batch0:{...}
                        batch1:{...}
                    },
                    1, env1:
                    {
                        batch0:{...}
                        batch1:{...}
                    }
                }
            }
        """
        self.files_envs_batches = []
        self.batch_count = 0
        for file_id, file in enumerate(self.dataset_files):
            env_batches = []
            for env_id in range(min(file['dones'].shape[1], self.config.num_env_import)):
                batch_count = math.ceil(file['dones'][:, env_id].shape[0] / self.config.batch_size)
                self.batch_count += batch_count
                env_batches.append((env_id, list(range(batch_count))))
            self.files_envs_batches.append((file_id, env_batches))

    def fill_batch(self):

        if len(self.files_envs_batches) == 0:
            return None

        batch = {}
        demo_batch = {}

        file_tuple = random.choice(self.files_envs_batches)
        file_rand = self.dataset_files[file_tuple[0]]
        env_rand_tuple = random.choice(file_tuple[1])
        env_rand = env_rand_tuple[0]
        batch_rand = random.choice(env_rand_tuple[1])

        env_rand_tuple[1].remove(batch_rand)
        if len(env_rand_tuple[1]) == 0:
            file_tuple[1].remove(env_rand_tuple)
            if len(file_tuple[1]) == 0:
                self.files_envs_batches.remove(file_tuple)

        batch_start = batch_rand * self.config.batch_size
        batch_end = min(batch_start + self.config.batch_size, file_rand['dones'].shape[0])
        indices = np.arange(batch_start, batch_end)
        global_rot = torch.from_numpy(file_rand['global_rot'][indices, env_rand, ...])
        root_pos = torch.from_numpy(file_rand['root_pos'][indices, env_rand, ...])
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
        dof_vel = torch.from_numpy(file_rand['dof_vel'][indices, env_rand, ...]).to(self.device)
        key_body_pos = sk_motion.global_translation[:, self.key_body_ids].to(self.device)
        actions = torch.from_numpy(file_rand['actions'][indices, env_rand, ...]).to(self.device)
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
        batch['disc_obs'] = self.make_with_hist_obs(disc_obs)
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
        batch['human_obs'] = human_obs
        batch['actions'] = actions
        batch['dones'] = torch.from_numpy(file_rand['dones'][indices, env_rand, ...]).to(self.device)
        batch['next_human_obs'] = torch.roll(human_obs, shifts=-1, dims=0)

        batch['future_obs'] = self.make_with_future_obs(disc_obs,hist_steps=2)
        batch['latents'], batch["mean"], batch["log_var"] = self.enc_sample(batch)
        batch['next_latents'] = torch.roll(batch['latents'], shifts=-1, dims=0)

        filled = 0
        demo_disc_obs = []
        demo_actions = []
        while filled < batch['dones'].shape[0]:
            file_rand = random.choice(self.demo_dataset_files)
            motion_end = min(batch['dones'].shape[0] - filled, file_rand['actions'].shape[0])
            disc_obs = torch.from_numpy(file_rand['disc_obs'][0:motion_end, 0, ...]).to(self.device)
            demo_disc_obs.append(self.make_with_hist_obs(disc_obs))
            demo_actions.append(actions)
            filled += motion_end

        demo_batch['disc_obs'] = torch.cat(demo_disc_obs, dim=0)

        key=list(batch.keys())[0]
        batch_indices = torch.randperm(batch[key].shape[0])
        for key in batch.keys():
            batch[key] = batch[key][batch_indices]

        key=list(demo_batch.keys())[0]
        demo_indices = torch.randperm(demo_batch[key].shape[0])
        for key in demo_batch.keys():
            demo_batch[key] = demo_batch[key][demo_indices]

        return batch, demo_batch

    def fit(self):
        start_epoch = self.current_epoch

        for self.current_epoch in range(start_epoch, self.config.max_epochs):
            print(f"Epoch: {self.current_epoch}")

            self.start_epoch()

            p_bar = tqdm(range(self.batch_count))

            v_loss_tensor = torch.zeros(self.batch_count)
            q_loss_tensor = torch.zeros(self.batch_count)
            desciptor_r = torch.zeros(self.batch_count)
            a_loss_tensor_adw_exp = torch.zeros(self.batch_count)
            a_loss_tensor_adw_neglog = torch.zeros(self.batch_count)
            a_loss_tensor_adw = torch.zeros(self.batch_count)
            enc_loss_log = torch.zeros(self.batch_count)
            disc_loss_log = torch.zeros(self.batch_count)

            batch_id = 0
            while (batch_tuple := self.fill_batch()) is not None:
                batch, demo_batch = batch_tuple

                desc_r = self.calculate_discriminator_reward(batch["disc_obs"]).squeeze()

                p_bar.update(1)
                p_bar.refresh()

                reward = desc_r.detach()

                """
                QF Loss
                """
                #print("Q step")
                q1_pred = self.qf1({"obs": batch["human_obs"], "actions": batch["actions"],
                     "latents": batch["latents"].detach()})
                q2_pred = self.qf2({"obs": batch["human_obs"], "actions": batch["actions"],
                     "latents": batch["latents"].detach()})
                target_vf_pred = self.vf({"obs": batch["next_human_obs"], "latents": batch['next_latents'].detach()}).detach()

                q_target = reward + (1. - batch['dones']) * self.discount * target_vf_pred
                q_target = q_target.detach()
                qf1_loss = self.qf_criterion(q1_pred, q_target)
                qf2_loss = self.qf_criterion(q2_pred, q_target)

                """
                VF Loss
                """
                #print("V step")
                q_pred = torch.min(
                    self.target_qf1({"obs": batch["human_obs"], "actions": batch["actions"],
                     "latents": batch["latents"].detach()}),
                    self.target_qf2({"obs": batch["human_obs"], "actions": batch["actions"],
                     "latents": batch["latents"].detach()}),
                ).detach()
                vf_pred = self.vf({"obs": batch["human_obs"], "latents": batch["latents"].detach()})
                vf_err = vf_pred - q_pred
                vf_sign = (vf_err > 0).float()
                vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
                vf_loss = (vf_weight * (vf_err ** 2)).mean()

                """
                Policy Loss
                """
                #print("P step")
                self.actor.training = True
                actor_out = self.actor.training_forward(
                    {"obs": batch["human_obs"],
                     "actions": batch["actions"],
                     "latents": batch["latents"].detach()})

                policy_logpp = -actor_out["neglogp"]

                adv = q_pred - vf_pred
                exp_adv = torch.exp(adv / self.beta)
                exp_adv = torch.clamp(exp_adv, max=100)

                weights = exp_adv.detach()  # exp_adv[:, 0].detach()
                actor_adw_loss = (-policy_logpp * weights).mean()

                enc_loss = self.enc_step(batch)

                disc_loss, disc_log_dict = self.discriminator_step({"AgentDiscObs": batch["disc_obs"],
                                                                    "DemoDiscObs": demo_batch["disc_obs"]})

                """
                Update networks
                """
                if batch_id % self.q_update_period == 0:
                    self.qf1_optimizer.zero_grad()
                    qf1_loss.backward()
                    self.qf1_optimizer.step()

                    self.qf2_optimizer.zero_grad()
                    qf2_loss.backward()
                    self.qf2_optimizer.step()

                    self.vf_optimizer.zero_grad()
                    vf_loss.backward()
                    self.vf_optimizer.step()

                if batch_id % self.policy_update_period == 0:

                    self.actor_optimizer.zero_grad()
                    actor_adw_loss.backward()
                    self.actor_optimizer.step()

                    self.encoder_optimizer.zero_grad()
                    self.decoder_optimizer.zero_grad()
                    enc_loss.backward()
                    self.encoder_optimizer.step()
                    self.decoder_optimizer.step()

                    self.discriminator_optimizer.zero_grad()
                    disc_loss.backward()
                    self.discriminator_optimizer.step()

                """
                Soft Updates
                """
                if batch_id % self.target_update_period == 0:
                    soft_update_from_to(
                        self.qf1, self.target_qf1, self.alpha
                    )
                    soft_update_from_to(
                        self.qf2, self.target_qf2, self.alpha
                    )

                """
                Log
                """
                desciptor_r[batch_id] = desc_r.mean().detach()

                desciptor_r[batch_id] = desc_r.mean().detach()

                q_loss_tensor[batch_id] = (qf1_loss.mean().detach() + qf2_loss.mean().detach()) / 2.

                v_loss_tensor[batch_id] = vf_loss.mean().detach()

                a_loss_tensor_adw[batch_id] = actor_adw_loss.mean().detach()
                enc_loss_log[batch_id] = enc_loss.mean().detach()
                disc_loss_log[batch_id] = disc_loss.mean().detach()
                a_loss_tensor_adw_exp[batch_id] = exp_adv.mean().detach()
                a_loss_tensor_adw_neglog[batch_id] = actor_out["neglogp"].mean().detach()

                batch_id += 1

            self.log_dict.update({
                "reward/desc": desciptor_r.mean(),
                "ac/v_loss": v_loss_tensor.mean(),
                "ac/q_loss": q_loss_tensor.mean(),
                "actor_loss/adw_exp": a_loss_tensor_adw_exp.mean(),
                "actor_loss/neg_log": a_loss_tensor_adw_neglog.mean(),
                "actor_loss/adw_after_clip": a_loss_tensor_adw.mean(),
                "actor_loss/enc_loss": enc_loss_log.mean(),
                "actor_loss/disc_loss": disc_loss_log.mean(),
            })

            self.fabric.log_dict(self.log_dict, self.current_epoch)

            if self.current_epoch % 30 == 0:
                self.save()

            del p_bar

    def fit_discriminator(self):
        start_epoch = self.current_epoch

        for self.current_epoch in range(start_epoch, self.config.max_epochs):
            print(f"Epoch: {self.current_epoch}")

            self.start_epoch()

            p_bar = tqdm(range(self.batch_count))

            disc_loss_log = torch.zeros(self.batch_count)

            batch_id = 0
            while (batch_tuple := self.fill_batch()) is not None:
                batch, demo_batch = batch_tuple

                p_bar.update(1)
                p_bar.refresh()

                disc_loss, disc_log_dict = self.discriminator_step({"AgentDiscObs": batch["disc_obs"],
                                                                    "DemoDiscObs": demo_batch["disc_obs"]})

                """
                Update networks
                """
                self.discriminator_optimizer.zero_grad()
                disc_loss.backward()
                self.discriminator_optimizer.step()

                """
                Log
                """
                disc_loss_log[batch_id] = disc_loss.mean().detach()

                batch_id += 1

            self.log_dict.update({
                "actor_loss/disc_loss": disc_loss_log.mean(),
            })

            self.fabric.log_dict(self.log_dict, self.current_epoch)

            if self.current_epoch % 1 == 0:
                self.save()

            del p_bar

    def fit_encoder(self):
        start_epoch = self.current_epoch

        for self.current_epoch in range(start_epoch, self.config.max_epochs):
            print(f"Epoch: {self.current_epoch}")

            self.start_epoch()

            p_bar = tqdm(range(self.batch_count))

            enc_loss_log = torch.zeros(self.batch_count)

            batch_id = 0
            while (batch_tuple := self.fill_batch()) is not None:
                batch, demo_batch = batch_tuple

                p_bar.update(1)
                p_bar.refresh()

                enc_loss = self.enc_step({"future_obs":batch["future_obs"], "latents":batch["latents"]})

                """
                Update networks
                """
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                enc_loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                """
                Log
                """

                enc_loss_log[batch_id] = enc_loss.mean().detach()

                batch_id += 1

            self.log_dict.update({
                "actor_loss/enc_loss": enc_loss_log.mean(),
            })

            self.fabric.log_dict(self.log_dict, self.current_epoch)

            if self.current_epoch % 1 == 0:
                self.save()

            del p_bar

    def discriminator_forward(self, obs: Tensor, return_norm_obs=False) -> Tensor:
        args = {"obs": obs}
        return self.discriminator(args, return_norm_obs=return_norm_obs)

    def enc_sample(self,batch):
        # Encoder forward pass
        latent_dist = self.encoder({"obs": batch["future_obs"]})  # Assuming input_obs contains encoder inputs

        # Split into mean and log variance
        mean, log_var = torch.chunk(latent_dist, 2, dim=-1)

        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        z = F.normalize(z, dim=-1)
        return z, mean, log_var

    def enc_step(self, batch):
        # Decoder forward pass
        decoded = self.decoder({"obs": batch["latents"]})

        # Calculate losses
        mse_loss = self.enc_loss_fn(decoded, batch["future_obs"])

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + batch["log_var"] - batch["mean"].pow(2) - batch["log_var"].exp(), dim=1).mean()

        return mse_loss + kl_loss

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

    # TODO:
    def make_with_future_obs(self, obs: torch.Tensor, flatten=True, hist_steps=None):
        """
        Constructs a tensor containing future observations for each time step.

        Args:
            obs (torch.Tensor): Tensor of shape (N_steps, ...).

        Returns:
            torch.Tensor: Tensor of shape (N_steps, hist_steps, ...) with future context,
                          or (N_steps, hist_steps * prod(obs.shape[1:])) if flattened.
        """
        N_steps = obs.shape[0]
        if hist_steps is None:
            hist_steps = self.all_config.env.config.discriminator_obs_historical_steps
        obs_shape = obs.shape[1:]

        # Pad at the end with zeros to allow future windowing beyond final step
        padding = torch.zeros((hist_steps - 1, *obs_shape), device=obs.device, dtype=obs.dtype)
        padded_obs = torch.cat([obs, padding], dim=0)  # Shape: (N_steps + hist_steps - 1, ...)

        # Create indices to extract future windows
        # todo: delte current obs only future
        indices = (torch.arange(N_steps, device=obs.device).unsqueeze(1) +
                   torch.arange(hist_steps, device=obs.device))  # Shape: (N_steps, hist_steps)

        # Gather the future slices
        temp = padded_obs[indices]  # Shape: (N_steps, hist_steps, ...)

        # Flatten the last two dimensions if needed
        return temp.reshape(N_steps, -1) if flatten else temp

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

    def get_state_dict(self, state_dict):
        extra_state_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.vf.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "target_qf1": self.target_qf1.state_dict(),
            "target_qf2": self.target_qf2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.vf_optimizer.state_dict(),
            "qf1_optimizer": self.qf1_optimizer.state_dict(),
            "qf2_optimizer": self.qf2_optimizer.state_dict(),
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

        state_dict["encoder"] = self.encoder.state_dict()
        state_dict["encoder_optimizer"] = self.encoder_optimizer.state_dict()

        state_dict["decoder"] = self.decoder.state_dict()
        state_dict["decoder_optimizer"] = self.decoder_optimizer.state_dict()

        state_dict["torch_rng_state"] = torch.get_rng_state()
        state_dict["random_rng_state"] = random.getstate()
        state_dict["numpy_rng_state"] = np.random.get_state()

        state_dict.update(extra_state_dict)
        return state_dict

    def save(self, path=None, name="last.ckpt", new_high_score=False):
        if path is None:
            path = self.fabric.loggers[0].log_dir
        root_dir = Path.cwd() / Path(self.fabric.loggers[0].root_dir)
        save_dir = Path.cwd() / Path(path)
        state_dict = self.get_state_dict({})
        #name = f"{name}_{self.current_epoch}"
        self.fabric.save(save_dir / name, state_dict)

    def load_encoder(self, checkpoint):
        if checkpoint is not None:
            checkpoint = Path(checkpoint).resolve()
            print(f"Loading encoder model from checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.encoder.load_state_dict(state_dict["encoder"])
            self.encoder_optimizer.load_state_dict(state_dict["encoder_optimizer"])

    def load_discriminator(self, checkpoint):
        if checkpoint is not None:
            checkpoint = Path(checkpoint).resolve()
            print(f"Loading encoder model from checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.discriminator.load_state_dict(state_dict["discriminator"])
            self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])

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
