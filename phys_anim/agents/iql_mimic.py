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
from phys_anim.envs.humanoid.humanoid_utils import build_disc_observations, build_disc_action_observations, \
    compute_humanoid_observations_max
import math
import numpy as np
import h5py



# def list_roll(inlist, n):
#    for i in range(n):
#        inlist.append(inlist.pop(0))

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class IQL_Mimic:

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

        self.dataset_file = h5py.File(self.all_config.algo.config.dataset_file, "r")

        self.dataset = {}

        self.dataset_len = self.dataset_file["obs"].shape[0]

        self.update_steps_per_stage = 1

        pass

    def fill_dataset(self):
        num_envs = self.dataset_file['obs'].shape[1]
        env_ind = random.randint(0, num_envs - 1)
        for attr in self.all_config.algo.config.attribs_to_import:
            self.dataset[attr] = torch.from_numpy(self.dataset_file[attr][:,env_ind]).to(self.device)


    def dataset_roll(self):
        for key in self.dataset.keys():
            self.dataset[key] = torch.roll(self.dataset[key], shifts=-self.config.batch_size)

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

        self._n_train_steps_total = 0
        self.q_update_period = 1
        self.policy_update_period = 1
        self.target_update_period = 1

        #state_dict = torch.load(Path.cwd() / "results/iql_mimic/lightning_logs/version_2/last.ckpt", map_location=self.device)
        #self.actor.load_state_dict(state_dict["actor"])
        #self.save(name="last_a.ckpt")

    def fit(self):

        for self.current_epoch in range(self.config.max_epochs):
            print(f"Epoch: {self.current_epoch}")
            batch_count = math.ceil(self.dataset_len / self.config.batch_size)

            self.fill_dataset()

            print(f'Value Step')

            v_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            q1_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            q2_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw_neglog = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw = torch.zeros(self.update_steps_per_stage * batch_count)
            for i in range(self.update_steps_per_stage):
                for batch_id in range(batch_count):

                    self.dataset_roll()

                    batch = {
                        "obs": self.dataset["obs"][0:self.config.batch_size],
                        "mimic_target_poses": self.dataset["mimic_target_poses"][0:self.config.batch_size],
                        "dones": self.dataset["dones"][0:self.config.batch_size],
                        "actions": self.dataset["actions"][0:self.config.batch_size],
                        "rewards": self.dataset["rewards"][0:self.config.batch_size],
                    }

                    rewards = batch["rewards"]

                    next_obs = torch.roll(batch["obs"], shifts=-1, dims=0)
                    next_targets = torch.roll(batch["mimic_target_poses"], shifts=-1, dims=0)

                    """
                    QF Loss
                    """
                    q1_pred = self.qf1({"obs": batch["obs"], "actions": batch["actions"],
                         "mimic_target_poses": batch["mimic_target_poses"]})
                    q2_pred = self.qf2({"obs": batch["obs"], "actions": batch["actions"],
                         "mimic_target_poses": batch["mimic_target_poses"]})
                    target_vf_pred = self.vf({"obs": next_obs, "mimic_target_poses": next_targets}).detach()

                    q_target = rewards + (1. - batch['dones']) * self.discount * target_vf_pred
                    q_target = q_target.detach()
                    qf1_loss = self.qf_criterion(q1_pred, q_target)
                    qf2_loss = self.qf_criterion(q2_pred, q_target)

                    """
                    VF Loss
                    """
                    q_pred = torch.min(
                        self.target_qf1({"obs": batch["obs"], "actions": batch["actions"],
                         "mimic_target_poses": batch["mimic_target_poses"]}),
                        self.target_qf2({"obs": batch["obs"], "actions": batch["actions"],
                         "mimic_target_poses": batch["mimic_target_poses"]}),
                    ).detach()
                    vf_pred = self.vf({"obs": batch["obs"], "mimic_target_poses": batch["mimic_target_poses"]})
                    vf_err = vf_pred - q_pred
                    vf_sign = (vf_err > 0).float()
                    vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
                    vf_loss = (vf_weight * (vf_err ** 2)).mean()

                    """
                    Policy Loss
                    """
                    self.actor.training = True
                    actor_out = self.actor.training_forward(
                        {"obs": batch["obs"],
                         "actions": batch["actions"],
                         "mimic_target_poses": batch["mimic_target_poses"]})

                    policy_logpp = -actor_out["neglogp"]

                    adv = q_pred - vf_pred
                    exp_adv = torch.exp(adv / self.beta)
                    exp_adv = torch.clamp(exp_adv, max=100)

                    weights = exp_adv.detach()#exp_adv[:, 0].detach()
                    policy_loss = (-policy_logpp * weights).mean()

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
                        policy_loss.backward()
                        self.actor_optimizer.step()

                    """
                    Soft Updates
                    """
                    if self._n_train_steps_total % self.target_update_period == 0:
                        soft_update_from_to(
                            self.qf1, self.target_qf1, self.soft_target_tau
                        )
                        soft_update_from_to(
                            self.qf2, self.target_qf2, self.soft_target_tau
                        )

                    self._n_train_steps_total += 1


                    q1_loss_tensor[batch_id * self.update_steps_per_stage + i] = qf1_loss.mean().detach()
                    q2_loss_tensor[batch_id * self.update_steps_per_stage + i] = qf2_loss.mean().detach()
                    v_loss_tensor[batch_id * self.update_steps_per_stage + i] = vf_loss.mean().detach()
                    a_loss_tensor_adw[batch_id * self.update_steps_per_stage + i] = policy_loss.mean().detach()
                    a_loss_tensor_adw_neglog[batch_id * self.update_steps_per_stage + i] = actor_out["neglogp"].mean().detach()


            self.log_dict.update({"ac/v_loss": v_loss_tensor.mean()})
            self.log_dict.update({"ac/q1_loss": q1_loss_tensor.mean()})
            self.log_dict.update({"ac/q2_loss": q2_loss_tensor.mean()})
            self.log_dict.update({"actor_loss/neg_log": a_loss_tensor_adw_neglog.mean()})
            self.log_dict.update({"actor_loss/loss_adw": a_loss_tensor_adw.mean()})

            self.fabric.log_dict(self.log_dict, self.current_epoch)

            if self.current_epoch % 10 == 0:
                self.save()

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
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.vf_optimizer.state_dict(),
            "epoch": self.current_epoch,
            "episode_reward_meter": None,
            "episode_length_meter": None,
            "best_evaluated_score": 0,
        }

        state_dict["running_val_norm"] = None

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
