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
from phys_anim.agents.models.actor import PPO_Actor
from hydra.utils import instantiate
from phys_anim.agents.models.common import NormObsBase
from phys_anim.envs.env_utils.general import StepTracker

import math
import h5py

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

        self.dataset_file = h5py.File(self.all_config.algo.config.dataset_file, "r")

        self.eval_dataset_file = h5py.File(self.all_config.algo.config.eval_dataset_file, "r")

        self.dataset = {}

        self.dataset_len = self.dataset_file["obs"].shape[0]

        self.update_steps_per_stage = 1

        pass

    def fill_dataset(self):
        num_envs = self.dataset_file['obs'].shape[1]
        env_ind = random.randint(0, num_envs - 1)
        for attr in self.all_config.algo.config.attribs_to_import:
            self.dataset[attr] = torch.from_numpy(self.dataset_file[attr][:,env_ind]).to(self.device)
        self.dataset['next_obs'] = torch.roll(self.dataset["obs"], shifts=-1, dims=0)
        self.dataset['next_targets'] = torch.roll(self.dataset["mimic_target_poses"], shifts=-1, dims=0)

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

        #state_dict = torch.load(Path.cwd() / "results/iql_mimic/lightning_logs/version_3/last.ckpt", map_location=self.device)
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
            q_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw_neglog = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            v_pred_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            q_pred_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            for i in range(self.update_steps_per_stage):
                for batch_id in range(batch_count):

                    indices = torch.randperm(len(self.dataset['obs']))[:self.config.batch_size]

                    batch = {
                        "obs": self.dataset["obs"][indices],
                        "mimic_target_poses": self.dataset["mimic_target_poses"][indices],
                        "dones": self.dataset["dones"][indices],
                        "actions": self.dataset["actions"][indices],
                        "rewards": self.dataset["rewards"][indices],
                        'next_obs': self.dataset["next_obs"][indices],
                        'next_targets': self.dataset["next_targets"][indices],
                    }

                    rewards = batch["rewards"]

                    next_obs = batch["next_obs"]
                    next_targets = batch["next_targets"]

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

                    qf_loss = (qf1_loss + qf2_loss) / 2

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
                    #vf_err = vf_pred - q_pred
                    #vf_sign = (vf_err > 0).float()
                    #vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
                    #vf_loss = (vf_weight * (vf_err ** 2)).mean()
                    vf_err = q_pred - vf_pred
                    vf_loss = torch.mean(torch.abs(self.expectile - (vf_err < 0).float()) * vf_err ** 2)

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
                        self.qf2_optimizer.zero_grad()
                        qf_loss.backward()
                        self.qf1_optimizer.step()
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
                            self.qf1, self.target_qf1, self.alpha
                        )
                        soft_update_from_to(
                            self.qf2, self.target_qf2, self.alpha
                        )

                    self._n_train_steps_total += 1


                    q1_loss_tensor[batch_id * self.update_steps_per_stage + i] = qf1_loss.mean().detach()
                    q2_loss_tensor[batch_id * self.update_steps_per_stage + i] = qf2_loss.mean().detach()
                    v_loss_tensor[batch_id * self.update_steps_per_stage + i] = vf_loss.mean().detach()
                    a_loss_tensor[batch_id * self.update_steps_per_stage + i] = policy_loss.mean().detach()
                    a_loss_tensor_adw_neglog[batch_id * self.update_steps_per_stage + i] = actor_out["neglogp"].mean().detach()
                    a_loss_tensor_adw[batch_id * self.update_steps_per_stage + i] = weights.mean().detach()
                    q_pred_tensor[batch_id * self.update_steps_per_stage + i] = q_pred.mean().detach()
                    v_pred_tensor[batch_id * self.update_steps_per_stage + i] = vf_pred.mean().detach()
                    q_loss_tensor[batch_id * self.update_steps_per_stage + i] = qf_loss.mean().detach()


            self.log_dict.update({"ac/v_loss": v_loss_tensor.mean()})
            self.log_dict.update({"ac/q1_loss": q1_loss_tensor.mean()})
            self.log_dict.update({"ac/q2_loss": q2_loss_tensor.mean()})
            self.log_dict.update({"ac/q_loss": q_loss_tensor.mean()})
            self.log_dict.update({"ac/q_pred": q_pred_tensor.mean()})
            self.log_dict.update({"ac/v_pred": v_pred_tensor.mean()})
            self.log_dict.update({"actor_loss/neg_log": a_loss_tensor_adw_neglog.mean()})
            self.log_dict.update({"actor_loss/adw": a_loss_tensor_adw.mean()})
            self.log_dict.update({"actor_loss/loss": a_loss_tensor.mean()})

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
            "q1": self.qf1.state_dict(),
            "q2": self.qf2.state_dict(),
            "q1_target": self.target_qf1.state_dict(),
            "q2_target": self.target_qf2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.vf_optimizer.state_dict(),
            "q1_optimizer": self.qf1_optimizer.state_dict(),
            "q2_optimizer": self.qf1_optimizer.state_dict(),
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

    def load(self, checkpoint: Path):
        if checkpoint is not None:
            checkpoint = Path(checkpoint).resolve()
            print(f"Loading model from checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.load_parameters(state_dict)

    def load_parameters(self, state_dict):
        self.current_epoch = state_dict["epoch"]
        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        self.actor.load_state_dict(state_dict["actor"])

        self.target_qf1.load_state_dict(state_dict["q1_target"])
        self.target_qf2.load_state_dict(state_dict["q1_target"])
        self.qf1.load_state_dict(state_dict["q1"])
        self.qf2.load_state_dict(state_dict["q2"])
        self.vf.load_state_dict(state_dict["critic"])

    def dataset_eval(self):
        print(self.dataset_file['obs'].shape)
        env_id = 0
        for step in range(self.dataset_file['obs'].shape[0]):
            batch = {
                'obs' : torch.from_numpy(self.dataset_file['obs'][step][env_id]).to(self.device).unsqueeze(0),
                'mimic_target_poses': torch.from_numpy(self.dataset_file['mimic_target_poses'][step][env_id]).to(self.device).unsqueeze(0),
                'actions': torch.from_numpy(self.dataset_file['actions'][step][env_id]).to(self.device).unsqueeze(0),
            }
            actor_out = self.actor.training_forward({"obs": batch["obs"], "actions": batch["actions"],
                                     "mimic_target_poses": batch["mimic_target_poses"]})
            vf = self.vf({"obs": batch["obs"], "mimic_target_poses": batch["mimic_target_poses"]})
            qf1 = self.qf1(
                {"obs": batch["obs"], "actions": batch["actions"],
                 "mimic_target_poses": batch["mimic_target_poses"]})
            qf2 = self.qf2(
                {"obs": batch["obs"], "actions": batch["actions"],
                 "mimic_target_poses": batch["mimic_target_poses"]})
            target_qf1 = self.target_qf1(
                {"obs": batch["obs"], "actions": batch["actions"],
                 "mimic_target_poses": batch["mimic_target_poses"]})
            target_qf2 = self.target_qf2(
                {"obs": batch["obs"], "actions": batch["actions"],
                 "mimic_target_poses": batch["mimic_target_poses"]})

            qf1_p = self.qf1(
                {"obs": batch["obs"], "actions": actor_out['mus'],
                 "mimic_target_poses": batch["mimic_target_poses"]})
            qf2_p = self.qf2(
                {"obs": batch["obs"], "actions": actor_out['mus'],
                 "mimic_target_poses": batch["mimic_target_poses"]})
            target_qf1_p = self.target_qf1(
                {"obs": batch["obs"], "actions": actor_out['mus'],
                 "mimic_target_poses": batch["mimic_target_poses"]})
            target_qf2_p = self.target_qf2(
                {"obs": batch["obs"], "actions": actor_out['mus'],
                 "mimic_target_poses": batch["mimic_target_poses"]})

            action_dif = torch.norm(batch['actions']-actor_out['mus'])

            q1_dif = qf1 - qf1_p
            q2_dif = qf2 - qf2_p

            # Generate action tensors that change gradually over steps
            action1 = torch.full_like(batch["actions"], step / self.dataset_file['obs'].shape[0])  # From 0 to 1
            action2 = torch.full_like(batch["actions"], -step / self.dataset_file['obs'].shape[0])  # From 0 to -1

            # Pass both to qf1
            qf1_action1 = self.qf1(
                {"obs": batch["obs"], "actions": action1, "mimic_target_poses": batch["mimic_target_poses"]})
            qf1_action2 = self.qf1(
                {"obs": batch["obs"], "actions": action2, "mimic_target_poses": batch["mimic_target_poses"]})

            # Compute the difference
            difference = qf1_action1 - qf1_action2

            log_dict = {
                "neglogp" : actor_out["neglogp"].mean(),
                "vf": vf,
                "qf1": qf1,
                "qf2": qf2,
                "target_qf1": target_qf1,
                "target_qf2": target_qf2,
                "qf1_p": qf1_p,
                "qf2_p": qf2_p,
                "target_qf1_p": target_qf1_p,
                "target_qf2_p": target_qf2_p,
                'mus_mean': actor_out['mus'].mean(),
                'actions_mean': batch['actions'].mean(),
                'action_dif': action_dif,
                'q1_dif': q1_dif,
                'q2_dif': q2_dif,
                'random_dif':difference
            }

            self.fabric.log_dict(log_dict, env_id*self.dataset_file['obs'].shape[1] + step)



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
