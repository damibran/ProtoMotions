import os.path
import random
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch import nn as nn
from torch import Tensor
from torch.cuda import device

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
from phys_anim.envs.humanoid.humanoid_utils import build_disc_observations,build_disc_action_observations, compute_humanoid_observations_max
import math
import numpy as np


# def list_roll(inlist, n):
#    for i in range(n):
#        inlist.append(inlist.pop(0))

class IQL_Calm:

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

        print("Demo Dataset Processing START")

        demo_motion_lib = MotionLib(
            motion_file=self.config.motion_file,
            dof_body_ids=self.dof_body_ids,
            dof_offsets=self.dof_offsets,
            key_body_ids=self.key_body_ids,
            device=self.device,
        )

        print("Demo Dataset Libs Loaded")
        self.demo_data = {}
        for motion_id in range(len(demo_motion_lib.state.motions)):
            motion_len = demo_motion_lib.get_motion_length(motion_id)
            dt = demo_motion_lib.get_motion(motion_id).time_delta

            if motion_len < 2.:
                print(f'skipped {demo_motion_lib.state.motion_files[motion_id]} with length {motion_len}')
                continue

            motion_times = torch.arange(0, motion_len, dt, device=self.device)
            state = demo_motion_lib.get_motion_state(motion_id, motion_times)

            self.demo_data[motion_id] = {
                "root_pos": state.root_pos,
                "root_rot": state.root_rot,
                "root_vel": state.root_vel,
                "root_ang_vel": state.root_ang_vel,
                "dof_pos": state.dof_pos,
                "dof_vel": state.dof_vel,
                "key_body_pos": state.key_body_pos,
                "DiscrimObs": build_disc_observations(
                    state.root_pos,
                    state.root_rot,
                    state.root_vel,
                    state.root_ang_vel,
                    state.dof_pos,
                    state.dof_vel,
                    state.key_body_pos,
                    torch.zeros(1, device=self.device),
                    self.all_config.env.config.humanoid_obs.local_root_obs,
                    self.all_config.env.config.humanoid_obs.root_height_obs,
                    self.all_config.robot.dof_obs_size,
                    self.dof_offsets,
                    False,
                    self.w_last,
                ),
                "HumanoidObservations": compute_humanoid_observations_max(
                    state.rb_pos,
                    state.rb_rot,
                    state.rb_vel,
                    state.rb_ang_vel,
                    torch.zeros(1, device=self.device),
                    self.all_config.env.config.humanoid_obs.local_root_obs,
                    self.all_config.env.config.humanoid_obs.root_height_obs,
                    self.w_last,
                )
            }

        #print("Demo Dataset Creating")
        #self.demo_dataset = {
        #    "root_pos": torch.zeros(motion_libs[0].actions.shape[0], state.root_pos.shape[1], device=self.device),
        #    "root_rot": torch.zeros(motion_libs[0].actions.shape[0], state.root_rot.shape[1], device=self.device),
        #    "root_vel": torch.zeros(motion_libs[0].actions.shape[0], state.root_vel.shape[1], device=self.device),
        #    "root_ang_vel": torch.zeros(motion_libs[0].actions.shape[0], state.root_ang_vel.shape[1],
        #                                device=self.device),
        #    "dof_pos": torch.zeros(motion_libs[0].actions.shape[0], state.dof_pos.shape[1], device=self.device),
        #    "dof_vel": torch.zeros(motion_libs[0].actions.shape[0], state.dof_vel.shape[1], device=self.device),
        #    "key_body_pos": torch.zeros(motion_libs[0].actions.shape[0], state.key_body_pos.shape[1],
        #                                state.key_body_pos.shape[2], device=self.device),
        #    "DiscrimObs": torch.zeros(motion_libs[0].actions.shape[0], self.discriminator_obs_size_per_step,
        #                              device=self.device),
        #    "Actions": torch.zeros(motion_libs[0].actions.shape[0], self.num_act, device=self.device),
        #    "HumanoidObservations": torch.zeros(motion_libs[0].actions.shape[0], self.num_obs, device=self.device)
        #}

        print("Dataset Processing START")
        motion_libs = []
        resets = []
        for m_file in self.config.dataset_files:
            motion_libs.append(StateActionLib(
                motion_file=m_file,
                dof_body_ids=self.dof_body_ids,
                dof_offsets=self.dof_offsets,
                key_body_ids=self.key_body_ids,
                device=self.device,
            ))
            motion_dir = os.path.dirname(m_file)
            motion_name = os.path.splitext(os.path.basename(m_file))[0]
            reset_file = motion_dir + "/" + motion_name + "_reset.npy"
            reset = torch.Tensor(np.load(reset_file)).to(self.device)
            resets.append(reset)

        print("Dataset Libs Loaded")
        self.data_subsets = []
        state = None
        for i in range(len(motion_libs)):
            lib = motion_libs[i]
            reset = resets[i]
            print(lib)
            lib_set = {}
            for motion_id in range(len(lib.state.motions)):
                motion_len = lib.get_motion_length(motion_id)
                dt = lib.get_motion(motion_id).time_delta

                motion_times = torch.arange(0, motion_len, dt, device=self.device)
                state = lib.get_motion_state(motion_id, motion_times)

                lib_set[motion_id] = {
                    "root_pos": state.root_pos,
                    "root_rot": state.root_rot,
                    "root_vel": state.root_vel,
                    "root_ang_vel": state.root_ang_vel,
                    "dof_pos": state.dof_pos,
                    "dof_vel": state.dof_vel,
                    "key_body_pos": state.key_body_pos,
                    "Actions": state.action,
                    "DiscrimObs": build_disc_action_observations(
                        state.root_pos,
                        state.root_rot,
                        state.root_vel,
                        state.root_ang_vel,
                        state.dof_pos,
                        state.dof_vel,
                        state.key_body_pos,
                        torch.zeros(1, device=self.device),
                        state.action,
                        self.all_config.env.config.humanoid_obs.local_root_obs,
                        self.all_config.env.config.humanoid_obs.root_height_obs,
                        self.all_config.robot.dof_obs_size,
                        self.dof_offsets,
                        False,
                        self.w_last,
                    ),
                    "HumanoidObservations": compute_humanoid_observations_max(
                        state.rb_pos,
                        state.rb_rot,
                        state.rb_vel,
                        state.rb_ang_vel,
                        torch.zeros(1, device=self.device),
                        self.all_config.env.config.humanoid_obs.local_root_obs,
                        self.all_config.env.config.humanoid_obs.root_height_obs,
                        self.w_last,
                    ),
                    "Reset": reset
                }
            self.data_subsets.append(lib_set)

        print("Dataset Creating")
        self.dataset = {
            "root_pos": torch.zeros(motion_libs[0].actions.shape[0], state.root_pos.shape[1], device=self.device),
            "root_rot": torch.zeros(motion_libs[0].actions.shape[0], state.root_rot.shape[1], device=self.device),
            "root_vel": torch.zeros(motion_libs[0].actions.shape[0], state.root_vel.shape[1], device=self.device),
            "root_ang_vel": torch.zeros(motion_libs[0].actions.shape[0], state.root_ang_vel.shape[1],
                                        device=self.device),
            "dof_pos": torch.zeros(motion_libs[0].actions.shape[0], state.dof_pos.shape[1], device=self.device),
            "dof_vel": torch.zeros(motion_libs[0].actions.shape[0], state.dof_vel.shape[1], device=self.device),
            "key_body_pos": torch.zeros(motion_libs[0].actions.shape[0], state.key_body_pos.shape[1],
                                        state.key_body_pos.shape[2], device=self.device),
            "DiscrimObs": torch.zeros(motion_libs[0].actions.shape[0], self.discriminator_obs_size_per_step,
                                      device=self.device),
            "Actions": torch.zeros(motion_libs[0].actions.shape[0], self.num_act, device=self.device),
            "HumanoidObservations": torch.zeros(motion_libs[0].actions.shape[0], self.num_obs, device=self.device),
            "Reset": torch.zeros(motion_libs[0].actions.shape[0], device=self.device)
        }

        self.dataset_len = motion_libs[0].actions.shape[0]

        self.update_steps_per_stage = 1

        pass

    def fill_dataset(self):
        #subset_ind = random.randint(0, len(self.demo_subsets)-1)
        #motion_ids = list(self.demo_subsets[subset_ind].keys())
        #ds_strt_ind = 0
        #while len(motion_ids) > 0:
        #    motion_id = random.choice(motion_ids)
        #    motion_ids.remove(motion_id)
        #    state = self.demo_subsets[subset_ind][motion_id]
        #    state_len = state["root_pos"].shape[0]
        #    self.demo_dataset["root_pos"][ds_strt_ind:ds_strt_ind + state_len] = state["root_pos"]
        #    self.demo_dataset["root_rot"][ds_strt_ind:ds_strt_ind + state_len] = state["root_rot"]
        #    self.demo_dataset["root_vel"][ds_strt_ind:ds_strt_ind + state_len] = state["root_vel"]
        #    self.demo_dataset["root_ang_vel"][ds_strt_ind:ds_strt_ind + state_len] = state["root_ang_vel"]
        #    self.demo_dataset["dof_pos"][ds_strt_ind:ds_strt_ind + state_len] = state["dof_pos"]
        #    self.demo_dataset["dof_vel"][ds_strt_ind:ds_strt_ind + state_len] = state["dof_vel"]
        #    self.demo_dataset["key_body_pos"][ds_strt_ind:ds_strt_ind + state_len] = state["key_body_pos"]
        #    self.demo_dataset["DiscrimObs"][ds_strt_ind:ds_strt_ind + state_len] = state["DiscrimObs"]
        #    self.demo_dataset["HumanoidObservations"][ds_strt_ind:ds_strt_ind + state_len] = state["HumanoidObservations"]
        #    self.demo_dataset["Actions"][ds_strt_ind:ds_strt_ind + state_len] = state["Actions"]
        #    ds_strt_ind += state_len

        subset_ind = random.randint(0, len(self.data_subsets)-1)
        motion_ids = list(self.data_subsets[subset_ind].keys())
        ds_strt_ind = 0
        while len(motion_ids) > 0:
            motion_id = random.choice(motion_ids)
            motion_ids.remove(motion_id)
            state = self.data_subsets[subset_ind][motion_id]
            state_len = state["root_pos"].shape[0]
            self.dataset["root_pos"][ds_strt_ind:ds_strt_ind + state_len] = state["root_pos"]
            self.dataset["root_rot"][ds_strt_ind:ds_strt_ind + state_len] = state["root_rot"]
            self.dataset["root_vel"][ds_strt_ind:ds_strt_ind + state_len] = state["root_vel"]
            self.dataset["root_ang_vel"][ds_strt_ind:ds_strt_ind + state_len] = state["root_ang_vel"]
            self.dataset["dof_pos"][ds_strt_ind:ds_strt_ind + state_len] = state["dof_pos"]
            self.dataset["dof_vel"][ds_strt_ind:ds_strt_ind + state_len] = state["dof_vel"]
            self.dataset["key_body_pos"][ds_strt_ind:ds_strt_ind + state_len] = state["key_body_pos"]
            self.dataset["HumanoidObservations"][ds_strt_ind:ds_strt_ind + state_len] = state["HumanoidObservations"]
            self.dataset["Actions"][ds_strt_ind:ds_strt_ind + state_len] = state["Actions"]
            self.dataset["Reset"][ds_strt_ind:ds_strt_ind + state_len] = state["Reset"]
            ds_strt_ind += state_len

    def dataset_roll(self):

        #self.demo_dataset["root_pos"] = torch.roll(self.demo_dataset["root_pos"], shifts=-self.config.batch_size)
        #self.demo_dataset["root_rot"] = torch.roll(self.demo_dataset["root_rot"], shifts=-self.config.batch_size)
        #self.demo_dataset["root_vel"] = torch.roll(self.demo_dataset["root_vel"], shifts=-self.config.batch_size)
        #self.demo_dataset["root_ang_vel"] = torch.roll(self.demo_dataset["root_ang_vel"], shifts=-self.config.batch_size)
        #self.demo_dataset["dof_pos"] = torch.roll(self.demo_dataset["dof_pos"], shifts=-self.config.batch_size)
        #self.demo_dataset["dof_vel"] = torch.roll(self.demo_dataset["dof_vel"], shifts=-self.config.batch_size)
        #self.demo_dataset["key_body_pos"] = torch.roll(self.demo_dataset["key_body_pos"], shifts=-self.config.batch_size)
        #self.demo_dataset["DiscrimObs"] = torch.roll(self.demo_dataset["DiscrimObs"], shifts=-self.config.batch_size)
        #self.demo_dataset["HumanoidObservations"] = torch.roll(self.demo_dataset["HumanoidObservations"], shifts=-self.config.batch_size)
        #self.demo_dataset["Actions"] = torch.roll(self.demo_dataset["Actions"], shifts=-self.config.batch_size)

        self.dataset["latents"] = torch.roll(self.dataset["latents"], shifts=-self.config.batch_size)
        self.dataset["root_pos"] = torch.roll(self.dataset["root_pos"], shifts=-self.config.batch_size)
        self.dataset["root_rot"] = torch.roll(self.dataset["root_rot"], shifts=-self.config.batch_size)
        self.dataset["root_vel"] = torch.roll(self.dataset["root_vel"], shifts=-self.config.batch_size)
        self.dataset["root_ang_vel"] = torch.roll(self.dataset["root_ang_vel"],shifts=-self.config.batch_size)
        self.dataset["dof_pos"] = torch.roll(self.dataset["dof_pos"], shifts=-self.config.batch_size)
        self.dataset["dof_vel"] = torch.roll(self.dataset["dof_vel"], shifts=-self.config.batch_size)
        self.dataset["key_body_pos"] = torch.roll(self.dataset["key_body_pos"],shifts=-self.config.batch_size)
        self.dataset["DiscrimObs"] = torch.roll(self.dataset["DiscrimObs"], shifts=-self.config.batch_size)
        self.dataset["HumanoidObservations"] = torch.roll(self.dataset["HumanoidObservations"],shifts=-self.config.batch_size)
        self.dataset["Actions"] = torch.roll(self.dataset["Actions"], shifts=-self.config.batch_size)
        self.dataset["Reset"] = torch.roll(self.dataset["Reset"], shifts=-self.config.batch_size)

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

        critic_s: NormObsBase = instantiate(
            self.config.critic_s, num_in=self.num_obs, num_out=1
        )
        critic_optimizer = instantiate(
            self.config.critic_optimizer,
            params=list(critic_s.parameters()),
        )
        self.critic_s, self.critic_s_optimizer = self.fabric.setup(critic_s, critic_optimizer)

        critic_sa: NormObsBase = instantiate(
            self.config.critic_sa, num_in=self.num_obs, num_out=1
        )
        critic_sa_optimizer = instantiate(
            self.config.critic_optimizer,
            params=list(critic_sa.parameters()),
        )
        self.critic_sa, self.critic_sa_optimizer = self.fabric.setup(critic_sa, critic_sa_optimizer)
        self.critic_sa_criterion = nn.MSELoss()

        target_critic: NormObsBase = instantiate(
            self.config.critic_sa, num_in=self.num_obs, num_out=1
        )

        self.target_critic = self.fabric.setup(target_critic)

        discriminator: JointDiscMLP = instantiate(
            self.config.discriminator,
            num_in=self.discriminator_obs_size_per_step
                   * self.all_config.env.config.discriminator_obs_historical_steps,
            num_out=1
        )
        discriminator_optimizer = instantiate(
            self.config.discriminator_optimizer,
            params=discriminator.parameters(),
        )

        self.discriminator, self.discriminator_optimizer = self.fabric.setup(
            discriminator, discriminator_optimizer
        )

        encoder: MLP_WithNorm = instantiate(
            self.config.encoder,
            num_in=self.discriminator_obs_size_per_step
                   * 60,
            num_out=sum(self.config.infomax_parameters.latent_dim)
        )
        encoder_optimizer = instantiate(
            self.config.discriminator_optimizer,
            params=discriminator.parameters(),
        )

        self.encoder, self.encoder_optimizer = self.fabric.setup(
            encoder, encoder_optimizer
        )

        state_dict = torch.load(Path.cwd() / "results/iql/lightning_logs/version_0/last.ckpt", map_location=self.device)
        self.actor.load_state_dict(state_dict["actor"])
        self.save(name="last_a.ckpt")

    def fit(self):

        for self.current_epoch in range(self.config.max_epochs):
            print(f"Epoch: {self.current_epoch}")
            batch_count = math.ceil(self.dataset_len / self.config.batch_size)

            self.fill_dataset()
            self.dataset["latents"] = self.sample_latents(self.dataset_len)

            print(f'Value Step')

            v_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            q_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            desciptor_r = torch.zeros(self.update_steps_per_stage * batch_count)
            for i in range(self.update_steps_per_stage):
                for batch_id in range(batch_count):

                    self.dataset_roll()

                    batch = {
                        "latents": self.dataset["latents"][0:self.config.batch_size].detach(),
                        "DiscrimObs": self.dataset["DiscrimObs"][0:self.config.batch_size],
                        "HumanoidObservations": self.dataset["HumanoidObservations"][0:self.config.batch_size],
                        "Actions": self.dataset["Actions"][0:self.config.batch_size],
                        "Reset": self.dataset["Reset"][0:self.config.batch_size],
                    }

                    desc_r = self.calculate_discriminator_reward({"obs":batch["DiscrimObs"],
                                                                  "latents":batch["latents"]}).squeeze()

                    reward = desc_r.detach()

                    desciptor_r[batch_id * self.update_steps_per_stage + i] = desc_r.mean().detach()

                    """
                    VF Loss
                    """
                    self.target_critic.train()
                    self.critic_s.train()
                    q_pred = self.target_critic(
                        {"obs": batch["HumanoidObservations"], "actions": batch["Actions"],
                         "latents": batch["latents"]}).detach()
                    vf_pred = self.critic_s({"obs": batch["HumanoidObservations"], "latents": batch["latents"]})
                    vf_err = vf_pred - q_pred
                    vf_sign = (vf_err > 0).float()
                    vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
                    value_loss = (vf_weight * (vf_err ** 2)).mean()

                    v_loss_tensor[batch_id * self.update_steps_per_stage + i] = value_loss.mean().detach()

                    """
                    QF Loss
                    """

                    next_obs = torch.roll(batch["HumanoidObservations"], shifts=-1, dims=0)
                    next_latents = torch.roll(batch["latents"], shifts=-1, dims=0)

                    q_target = reward + (1. - batch["Reset"]) * self.discount * self.critic_s({"obs": next_obs, "latents": next_latents}).detach()
                    q_target = q_target.detach()
                    q_pred = self.critic_sa({"obs": batch["HumanoidObservations"], "actions": batch["Actions"],
                                             "latents": batch["latents"]})

                    q_loss = self.critic_sa_criterion(q_target, q_pred)

                    q_loss_tensor[batch_id * self.update_steps_per_stage + i] = q_loss.mean().detach()

                    """
                    Step
                    """

                    self.critic_s_optimizer.zero_grad()
                    self.fabric.backward(value_loss.mean())
                    self.critic_s_optimizer.step()

                    self.critic_sa_optimizer.zero_grad()
                    self.fabric.backward(q_loss.mean())
                    self.critic_sa_optimizer.step()

                    for param_cur, param_target in zip(self.critic_sa.parameters(), self.target_critic.parameters()):
                        param_target.data = (1 - self.alpha) * param_target.data + self.alpha * param_cur.data

            self.log_dict.update({
                "reward/desc":desciptor_r.mean(),
            })
            self.log_dict.update({"ac/v_loss": v_loss_tensor.mean()})
            self.log_dict.update({"ac/q_loss": q_loss_tensor.mean()})

            a_loss_tensor_adw_exp = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw_neglog = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw = torch.zeros(self.update_steps_per_stage * batch_count)
            enc_loss_log = torch.zeros(self.update_steps_per_stage * batch_count)
            disc_loss_log = torch.zeros(self.update_steps_per_stage * batch_count)

            print(f'Actor Step')
            for i in range(self.update_steps_per_stage):
                for batch_id in range(batch_count):
                    self.dataset_roll()

                    batch = {
                        "latents": self.dataset["latents"][0:self.config.batch_size],
                        "DiscrimObs": self.dataset["DiscrimObs"][0:self.config.batch_size],
                        "HumanoidObservations": self.dataset["HumanoidObservations"][0:self.config.batch_size],
                        "Actions": self.dataset["Actions"][0:self.config.batch_size]
                    }

                    self.actor.training = True
                    actor_out = self.actor.training_forward(
                        {"obs": batch["HumanoidObservations"],
                         "actions": batch["Actions"],
                         "latents": batch["latents"]})

                    self.target_critic.eval()
                    self.critic_s.eval()
                    q_val = self.target_critic(
                        {"obs": batch["HumanoidObservations"], "actions": batch["Actions"],
                         "latents": batch["latents"]})
                    v_val = self.critic_s({"obs": batch["HumanoidObservations"], "latents": batch["latents"]})

                    adv = q_val.detach() - v_val.detach()
                    exp_adv = torch.exp(adv / self.beta)
                    exp_adv = torch.clamp(exp_adv, max=100)

                    actor_adw_loss = (exp_adv * actor_out["neglogp"]).mean()

                    enc_loss = self.enc_reg_loss(self.config.batch_size)

                    disc_loss = self.conditional_disc_loss({"obs":batch["DiscrimObs"],"latents":batch["latents"].detach()})

                    loss = actor_adw_loss + enc_loss + disc_loss

                    self.actor_optimizer.zero_grad()
                    self.encoder_optimizer.zero_grad()
                    self.discriminator_optimizer.zero_grad()
                    self.fabric.backward(loss.mean(), retain_graph=True)
                    self.actor_optimizer.step()
                    self.encoder_optimizer.step()
                    self.discriminator_optimizer.step()

                    a_loss_tensor_adw_exp[batch_id * self.update_steps_per_stage + i] = exp_adv.mean().detach()
                    a_loss_tensor_adw_neglog[batch_id * self.update_steps_per_stage + i] = actor_out[
                        "neglogp"].mean().detach()
                    a_loss_tensor_adw[batch_id * self.update_steps_per_stage + i] = actor_adw_loss.mean().detach()
                    enc_loss_log[batch_id * self.update_steps_per_stage + i] = enc_loss.mean().detach()
                    disc_loss_log[batch_id * self.update_steps_per_stage + i] = disc_loss.mean().detach()


            self.log_dict.update({"actor_loss/adw_exp": a_loss_tensor_adw_exp.mean()})
            self.log_dict.update({"actor_loss/neg_log": a_loss_tensor_adw_neglog.mean()})
            self.log_dict.update({"actor_loss/loss_adw": a_loss_tensor_adw.mean()})
            self.log_dict.update({"actor_loss/enc_loss": enc_loss_log.mean()})
            self.log_dict.update({"actor_loss/disc_loss": disc_loss_log.mean()})

            '''
            for i in range(self.update_steps_per_stage):
                for batch_id in range(batch_count):

                    self.dataset_roll()

                    batch = {
                        "latents": self.dataset["latents"][0:self.config.batch_size],
                        "root_pos" : self.dataset["root_pos"][0:self.config.batch_size],
                        "root_rot": self.dataset["root_rot"][0:self.config.batch_size],
                        "root_vel": self.dataset["root_vel"][0:self.config.batch_size],
                        "root_ang_vel": self.dataset["root_ang_vel"][0:self.config.batch_size],
                        "dof_pos": self.dataset["dof_pos"][0:self.config.batch_size],
                        "dof_vel": self.dataset["dof_vel"][0:self.config.batch_size],
                        "dof_vel": self.dataset["dof_vel"][0:self.config.batch_size],
                        "key_body_pos": self.dataset["key_body_pos"][0:self.config.batch_size],
                        "DiscrimObs": self.dataset["DiscrimObs"][0:self.config.batch_size],
                        "HumanoidObservations": self.dataset["HumanoidObservations"][0:self.config.batch_size],
                        "Actions": self.dataset["Actions"][0:self.config.batch_size]
                    }

                    demo_batch = {
                        "root_pos": self.demo_dataset["root_pos"][0:self.config.batch_size],
                        "root_rot": self.demo_dataset["root_rot"][0:self.config.batch_size],
                        "root_vel": self.demo_dataset["root_vel"][0:self.config.batch_size],
                        "root_ang_vel": self.demo_dataset["root_ang_vel"][0:self.config.batch_size],
                        "dof_pos": self.demo_dataset["dof_pos"][0:self.config.batch_size],
                        "dof_vel": self.demo_dataset["dof_vel"][0:self.config.batch_size],
                        "dof_vel": self.demo_dataset["dof_vel"][0:self.config.batch_size],
                        "key_body_pos": self.demo_dataset["key_body_pos"][0:self.config.batch_size],
                        "DiscrimObs": self.demo_dataset["DiscrimObs"][0:self.config.batch_size],
                        "HumanoidObservations": self.demo_dataset["HumanoidObservations"][0:self.config.batch_size],
                        "Actions": self.demo_dataset["Actions"][0:self.config.batch_size]
                    }

                    self.actor.training = False
                    with torch.no_grad():
                        actor_eval_out = self.actor.eval_forward(
                            {"obs": batch["HumanoidObservations"], "latents": batch["latents"]})

                    agent_disc_obs = build_disc_action_observations(
                        batch["root_pos"],
                        batch["root_rot"],
                        batch["root_vel"],
                        batch["root_ang_vel"],
                        batch["dof_pos"],
                        batch["dof_vel"],
                        batch["key_body_pos"],
                        torch.zeros(1, device=self.device),
                        actor_eval_out["actions"],
                        self.all_config.env.config.humanoid_obs.local_root_obs,
                        self.all_config.env.config.humanoid_obs.root_height_obs,
                        self.all_config.robot.dof_obs_size,
                        self.dof_offsets,
                        False,
                        self.w_last,
                    )

                    disc_loss, disc_log_dict = self.encoder_step(
                        {"AgentDiscObs": agent_disc_obs, "DemoDiscObs": demo_batch["DiscrimObs"], "latents": batch["latents"]})
                    self.log_dict.update(disc_log_dict)

                    self.discriminator_optimizer.zero_grad()
                    self.fabric.backward(disc_loss)
                    self.discriminator_optimizer.step()'''

            self.fabric.log_dict(self.log_dict, self.current_epoch)

            if self.current_epoch % 10 == 0:
                self.save()

    # todo: make faster
    def sample_latents(self, n):
        enc_in = self.sample_enc_demo_obs(n)

        latents = self.encoder({"obs": enc_in})

        return latents

    def sample_enc_demo_obs(self, n):
        motion_ids = np.random.choice(np.array(list(self.demo_data.keys())), n)
        result = []
        for m_id in motion_ids:
            len = self.demo_data[m_id]["DiscrimObs"].shape[0]
            truncated_len = len - self.config.num_obs_enc_steps

            assert truncated_len >= 0

            start = random.randint(0, truncated_len)

            enc_in = self.demo_data[m_id]["DiscrimObs"][start: start + self.config.num_obs_enc_steps].flatten()
            result.append(enc_in)

        return torch.stack(result)

    def sample_enc_demo_obs_pair(self, n):
        motion_ids = np.random.choice(np.array(list(self.demo_data.keys())), n)
        result1 = []
        result2 = []
        for m_id in motion_ids:
            len = self.demo_data[m_id]["DiscrimObs"].shape[0]
            truncated_len = len - self.config.num_obs_enc_steps

            assert truncated_len >= 0

            start1 = random.randint(0, truncated_len)
            start2 = random.randint(start1, min(start1 + self.config.num_obs_enc_steps // 2, truncated_len))

            enc_in1 = self.demo_data[m_id]["DiscrimObs"][start1: start1 + self.config.num_obs_enc_steps].flatten()
            enc_in2 = self.demo_data[m_id]["DiscrimObs"][start2: start2 + self.config.num_obs_enc_steps].flatten()

            result1.append(enc_in1)
            result2.append(enc_in2)

        return torch.stack(result1), torch.stack(result2)

    def calculate_discriminator_reward(self, input_dict):
        """
        input dict = {
            "obs":
            "latents"
        }
        """

        with torch.no_grad():
            disc_logits = self.discriminator(input_dict)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self.config.discriminator_reward_w

        return disc_r

    def enc_reg_loss(self, b_size):
        enc_amp_obs_demo = self.sample_enc_demo_obs(b_size)

        amp_obs_encoding = self.encoder({"obs": enc_amp_obs_demo})

        # Loss for uniform distribution over the sphere
        def uniform_loss(x, t=2):
            return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
        uniform_l = uniform_loss(amp_obs_encoding)

        similar_enc_amp_obs_demo0, similar_enc_amp_obs_demo1 = self.sample_enc_demo_obs_pair(b_size)

        similar_amp_obs_encoding0 = self.encoder({"obs": similar_enc_amp_obs_demo0})
        similar_amp_obs_encoding1 = self.encoder({"obs": similar_enc_amp_obs_demo1})

        # Loss for alignment - overlapping motions should have 'close' embeddings
        def align_loss(x, y, alpha=2):
            return torch.linalg.norm(x - y, ord=2, dim=1).pow(alpha).mean()
        align_l = align_loss(similar_amp_obs_encoding0, similar_amp_obs_encoding1)

        loss = align_l + 0.5 * uniform_l

        return loss

    def conditional_disc_loss(self, input_dict):
        """
        input dict = {
            "obs":
            "latents"
        }
        """

        batch_size = input_dict["obs"].shape[0]

        disc_agent_logit = self.discriminator(input_dict)

        demo_enc_obs = self.sample_enc_demo_obs(batch_size)
        reshaped_obs = demo_enc_obs.view(batch_size, self.config.num_obs_enc_steps, self.discriminator_obs_size_per_step)
        random_indices = torch.randint(0, self.config.num_obs_enc_steps, (batch_size,))
        demo_disc_obs = reshaped_obs[torch.arange(batch_size), random_indices]

        demo_disc_obs.requires_grad = True

        with torch.no_grad():
            demo_enc = self.encoder({"obs": demo_enc_obs})
        demo_enc.requires_grad = True

        demo_dict =  self.discriminator({"obs":demo_disc_obs,"latents":demo_enc}, return_norm_obs=True)
        disc_demo_logit = demo_dict["outs"]
        demo_norm_obs = demo_dict["norm_obs"]

        # prediction loss
        disc_loss_agent = self.disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self.disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.discriminator.logit_weights()
        disc_logit_loss = sum([p.pow(2).sum() for p in logit_weights])
        disc_loss += self.config.discriminator_logit_weight_decay * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, (demo_norm_obs, demo_enc),
                                             grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self.config.discriminator_grad_penalty * disc_grad_penalty

        # weight decay
        if self.config.discriminator_weight_decay != 0:
            all_weight_params = self.discriminator.all_discriminator_weights()
            total: Tensor = sum([p.pow(2).sum() for p in all_weight_params])
            disc_loss += total * self.config.discriminator_weight_decay

        return disc_loss

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
            "critic": self.critic_s.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_s_optimizer.state_dict(),
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
