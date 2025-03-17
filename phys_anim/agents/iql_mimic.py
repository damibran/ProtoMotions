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


# def list_roll(inlist, n):
#    for i in range(n):
#        inlist.append(inlist.pop(0))

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

        self.subsets = []

        for _dir in self.all_config.algo.config.dataset_dirs:
            subset = {}
            for attr in self.all_config.algo.config.attribs_to_import:
                file_path = os.path.join(_dir, self.all_config.motion_name+'_'+attr+'.npy')
                loaded = torch.from_numpy(np.load(file_path)).to(self.device)
                subset[attr] = loaded

            for attr in subset.keys():
                if attr == "actions" or attr == "rew_buf":
                    # Shift tensor elements to the left and trim size
                    subset[attr] = torch.roll(subset[attr], shifts=-1, dims=0)[:-1]
                else:
                    # Remove the last entry for other tensors
                    subset[attr] = subset[attr][:-1]

            self.subsets.append(subset)

        print("Dataset Creating")
        self.dataset={}
        for attr in self.all_config.algo.config.attribs_to_import:
            self.dataset[attr] = self.subsets[0][attr]

        self.dataset_len = self.dataset["obs_buf"].shape[0]

        self.update_steps_per_stage = 1

        pass

    def fill_dataset(self):
        subset_ind = random.randint(0, len(self.subsets) - 1)
        for attr in self.all_config.algo.config.attribs_to_import:
            self.dataset[attr] = self.subsets[subset_ind][attr]


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

        # state_dict = torch.load(Path.cwd() / "results/iql/lightning_logs/version_0/last.ckpt", map_location=self.device)
        # self.actor.load_state_dict(state_dict["actor"])
        # self.save(name="last_a.ckpt")

    def fit(self):

        for self.current_epoch in range(self.config.max_epochs):
            print(f"Epoch: {self.current_epoch}")
            batch_count = math.ceil(self.dataset_len / self.config.batch_size)

            self.fill_dataset()

            print(f'Value Step')

            v_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            q_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            for i in range(self.update_steps_per_stage):
                for batch_id in range(batch_count):

                    self.dataset_roll()

                    batch = {
                        "obs_buf": self.dataset["obs_buf"][0:self.config.batch_size],
                        "mimic_target_poses": self.dataset["mimic_target_poses"][0:self.config.batch_size],
                        "reset_buf": self.dataset["reset_buf"][0:self.config.batch_size],
                        "actions": self.dataset["actions"][0:self.config.batch_size],
                        "rewards": self.dataset["rew_buf"][0:self.config.batch_size],
                    }

                    reward = batch["rewards"]

                    """
                    VF Loss
                    """
                    self.target_critic.train()
                    self.critic_s.train()
                    q_pred = self.target_critic(
                        {"obs": batch["obs_buf"], "actions": batch["actions"],
                         "mimic_target_poses": batch["mimic_target_poses"]}).detach()
                    vf_pred = self.critic_s({"obs": batch["obs_buf"], "mimic_target_poses": batch["mimic_target_poses"]})
                    vf_err = vf_pred - q_pred
                    vf_sign = (vf_err > 0).float()
                    vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
                    value_loss = (vf_weight * (vf_err ** 2)).mean()

                    v_loss_tensor[batch_id * self.update_steps_per_stage + i] = value_loss.mean().detach()

                    """
                    QF Loss
                    """

                    next_obs = torch.roll(batch["obs_buf"], shifts=-1, dims=0)
                    next_targets = torch.roll(batch["mimic_target_poses"], shifts=-1, dims=0)

                    q_target = reward + (1. - batch["reset_buf"]) * self.discount * self.critic_s(
                        {"obs": next_obs, "mimic_target_poses": next_targets}).detach()
                    q_target = q_target.detach()
                    q_pred = self.critic_sa({"obs": batch["obs_buf"], "actions": batch["actions"],
                                             "mimic_target_poses": batch["mimic_target_poses"]})

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

            self.log_dict.update({"ac/v_loss": v_loss_tensor.mean()})
            self.log_dict.update({"ac/q_loss": q_loss_tensor.mean()})

            a_loss_tensor_adw_exp = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw_neglog = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw = torch.zeros(self.update_steps_per_stage * batch_count)

            print(f'Actor Step')
            for i in range(self.update_steps_per_stage):
                for batch_id in range(batch_count):
                    self.dataset_roll()

                    batch = {
                        "obs_buf": self.dataset["obs_buf"][0:self.config.batch_size],
                        "mimic_target_poses": self.dataset["mimic_target_poses"][0:self.config.batch_size],
                        "reset_buf": self.dataset["reset_buf"][0:self.config.batch_size],
                        "actions": self.dataset["actions"][0:self.config.batch_size],
                        "rewards": self.dataset["rew_buf"][0:self.config.batch_size],
                    }

                    self.actor.training = True
                    actor_out = self.actor.training_forward(
                        {"obs": batch["obs_buf"],
                         "actions": batch["actions"],
                         "mimic_target_poses": batch["mimic_target_poses"]})

                    self.target_critic.eval()
                    self.critic_s.eval()
                    q_val = self.target_critic(
                        {"obs": batch["obs_buf"], "actions": batch["actions"],
                         "mimic_target_poses": batch["mimic_target_poses"]})
                    v_val = self.critic_s({"obs": batch["obs_buf"], "mimic_target_poses": batch["mimic_target_poses"]})

                    adv = q_val.detach() - v_val.detach()
                    exp_adv = torch.exp(adv / self.beta)
                    exp_adv = torch.clamp(exp_adv, max=100)

                    actor_adw_loss = (exp_adv * actor_out["neglogp"]).mean()

                    loss = actor_adw_loss

                    self.actor_optimizer.zero_grad()
                    self.fabric.backward(loss.mean(), retain_graph=True)
                    self.actor_optimizer.step()

                    a_loss_tensor_adw_exp[batch_id * self.update_steps_per_stage + i] = exp_adv.mean().detach()
                    a_loss_tensor_adw_neglog[batch_id * self.update_steps_per_stage + i] = actor_out[
                        "neglogp"].mean().detach()
                    a_loss_tensor_adw[batch_id * self.update_steps_per_stage + i] = actor_adw_loss.mean().detach()

            self.log_dict.update({"actor_loss/adw_exp": a_loss_tensor_adw_exp.mean()})
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
            "critic": self.critic_s.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_s_optimizer.state_dict(),
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
