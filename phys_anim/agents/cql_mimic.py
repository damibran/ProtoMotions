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

            # Zero out rew_buf elements where reset_buf is 1
            reset_buf_inverted = 1 - subset["reset_buf"]  # Invert reset_buf (0 becomes 1, 1 becomes 0)
            subset["rew_buf"] = subset["rew_buf"] * reset_buf_inverted  # Zero out where reset_buf is 1

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
        self.target_qf1 = self.fabric.setup(target_qf1)

        target_qf2: NormObsBase = instantiate(
            self.config.critic_sa, num_in=self.num_obs, num_out=1
        )
        self.target_qf2 = self.fabric.setup(target_qf2)

        vf: NormObsBase = instantiate(
            self.config.critic_s, num_in=self.num_obs, num_out=1
        )
        vf_optimizer = instantiate(
            self.config.critic_optimizer,
            params=list(vf.parameters()),
        )
        self.vf, self.vf_optimizer = self.fabric.setup(vf, vf_optimizer)

        target_vf: NormObsBase = instantiate(
            self.config.critic_s, num_in=self.num_obs, num_out=1
        )
        self.target_vf = self.fabric.setup(target_vf)

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self._n_train_steps_total = 0
        self.q_update_period = 1
        self.policy_update_period = 1
        self.target_update_period = 1
        self.soft_target_tau = 1

        self.use_automatic_entropy_tuning = True
        target_entropy = None
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.num_act).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=self.all_config.algo.config.actor_optimizer.lr,
            )

        lagrange_thresh = 10.0
        self.with_lagrange = True
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = torch.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = torch.optim.Adam(
                [self.log_alpha_prime],
                lr=self.all_config.algo.config.critic_optimizer.lr,
            )

        self.num_qs = 2

        ## min Q
        self.temp = 1.0
        self.min_q_version = 3
        self.min_q_weight = 1.0

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

        self.max_q_backup = False
        self.deterministic_backup = True
        self.num_random = 10

        self.policy_eval_start = 10000

        self._current_epoch = 0

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
            q1_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            q2_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw_neglog = torch.zeros(self.update_steps_per_stage * batch_count)
            a_loss_tensor_adw = torch.zeros(self.update_steps_per_stage * batch_count)
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

                    rewards = batch['rewards']
                    terminals = batch['reset_buf']
                    actions = batch['actions']

                    next_obs = torch.roll(batch["obs_buf"], shifts=-1, dims=0)
                    next_targets = torch.roll(batch["mimic_target_poses"], shifts=-1, dims=0)

                    """
                    Policy and Alpha Loss
                    """
                    # new_obs_actions, policy_mean, policy_log_std, log_pi, *_
                    actor_out = self.actor.eval_forward(
                        {"obs":batch["obs_buf"], "mimic_target_poses": batch["mimic_target_poses"]}
                    )

                    new_obs_actions = actor_out["actions"]
                    log_pi = -actor_out["neglogp"] #todo: should it be negative or what?

                    if self.use_automatic_entropy_tuning:
                        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                        self.alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer.step()
                        alpha = self.log_alpha.exp()
                    else:
                        alpha_loss = 0
                        alpha = 1

                    if self.num_qs == 1:
                        #q_new_actions = self.qf1(obs, new_obs_actions)
                        pass
                    else:
                        q_new_actions = torch.min(
                            self.qf1({"obs": batch["obs_buf"], "actions": new_obs_actions,
                         "mimic_target_poses": batch["mimic_target_poses"]}),
                            self.qf2({"obs": batch["obs_buf"], "actions": new_obs_actions,
                         "mimic_target_poses": batch["mimic_target_poses"]}),
                        )

                    policy_loss = (alpha * log_pi - q_new_actions).mean()

                    if self._current_epoch < self.policy_eval_start:
                        """
                        For the initial few epochs, try doing behaivoral cloning, if needed
                        conventionally, there's not much difference in performance with having 20k 
                        gradient steps here, or not having it
                        """
                        actor_out = self.actor.training_forward({"obs": batch["obs_buf"], "actions": actions,
                         "mimic_target_poses": batch["mimic_target_poses"]})
                        policy_log_prob = -actor_out["neglogp"]
                        policy_loss = (alpha * log_pi - policy_log_prob).mean()

                    """
                    QF Loss
                    """
                    q1_pred = self.qf1({"obs": batch["obs_buf"], "actions": actions,
                         "mimic_target_poses": batch["mimic_target_poses"]})
                    if self.num_qs > 1:
                        q2_pred = self.qf2({"obs": batch["obs_buf"], "actions": actions,
                         "mimic_target_poses": batch["mimic_target_poses"]})

                    actor_out = self.actor.eval_forward(
                        {"obs":next_obs, "mimic_target_poses": next_targets}
                    )
                    new_next_actions = actor_out['actions']
                    new_log_pi = -actor_out['neglogp']
                    actor_out_cur = self.actor.eval_forward(
                        {"obs":batch["obs_buf"], "mimic_target_poses": batch["mimic_target_poses"]}
                    )
                    new_curr_actions = actor_out_cur['actions']
                    new_curr_log_pi = -actor_out_cur['neglogp']

                    if not self.max_q_backup:
                        if self.num_qs == 1:
                            target_q_values = self.target_qf1({"obs":next_obs, "mimic_target_poses": next_targets, 'actions':new_next_actions})
                        else:
                            target_q_values = torch.min(
                                self.target_qf1({"obs":next_obs, "mimic_target_poses": next_targets, 'actions':new_next_actions}),
                                self.target_qf2({"obs":next_obs, "mimic_target_poses": next_targets, 'actions':new_next_actions}),
                            )

                        if not self.deterministic_backup:
                            target_q_values = target_q_values - alpha * new_log_pi

                    #if self.max_q_backup: #False
                    #    """when using max q backup"""
                    #    next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
                    #    target_qf1_values = \
                    #    self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1,
                    #                                                                                                 1)
                    #    target_qf2_values = \
                    #    self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1,
                    #                                                                                                 1)
                    #    target_q_values = torch.min(target_qf1_values, target_qf2_values)

                    q_target = rewards + (1. - terminals) * self.discount * target_q_values
                    q_target = q_target.detach()

                    qf1_loss = self.qf_criterion(q1_pred, q_target)
                    if self.num_qs > 1:
                        qf2_loss = self.qf_criterion(q2_pred, q_target)

                    ## add CQL
                    random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random,
                                                              actions.shape[-1]).uniform_(-1, 1)  # .cuda()
                    curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random,
                                                                                 network=self.policy)
                    new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs,
                                                                                    num_actions=self.num_random,
                                                                                    network=self.policy)
                    q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
                    q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
                    q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
                    q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
                    q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
                    q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)

                    cat_q1 = torch.cat(
                        [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
                    )
                    cat_q2 = torch.cat(
                        [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
                    )
                    std_q1 = torch.std(cat_q1, dim=1)
                    std_q2 = torch.std(cat_q2, dim=1)

                    if self.min_q_version == 3:
                        # importance sammpled version
                        random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
                        cat_q1 = torch.cat(
                            [q1_rand - random_density, q1_next_actions - new_log_pis.detach(),
                             q1_curr_actions - curr_log_pis.detach()], 1
                        )
                        cat_q2 = torch.cat(
                            [q2_rand - random_density, q2_next_actions - new_log_pis.detach(),
                             q2_curr_actions - curr_log_pis.detach()], 1
                        )

                    min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp
                    min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp

                    """Subtract the log likelihood of data"""
                    min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
                    min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

                    if self.with_lagrange:
                        alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
                        min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
                        min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

                        self.alpha_prime_optimizer.zero_grad()
                        alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
                        alpha_prime_loss.backward(retain_graph=True)
                        self.alpha_prime_optimizer.step()

                    qf1_loss = qf1_loss + min_qf1_loss
                    qf2_loss = qf2_loss + min_qf2_loss

                    """
                    Update networks
                    """
                    # Update the Q-functions iff
                    self._num_q_update_steps += 1
                    self.qf1_optimizer.zero_grad()
                    qf1_loss.backward(retain_graph=True)
                    self.qf1_optimizer.step()

                    if self.num_qs > 1:
                        self.qf2_optimizer.zero_grad()
                        qf2_loss.backward(retain_graph=True)
                        self.qf2_optimizer.step()

                    self._num_policy_update_steps += 1
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward(retain_graph=False)
                    self.policy_optimizer.step()

                    """
                    Soft Updates
                    """
                    ptu.soft_update_from_to(
                        self.qf1, self.target_qf1, self.soft_target_tau
                    )
                    if self.num_qs > 1:
                        ptu.soft_update_from_to(
                            self.qf2, self.target_qf2, self.soft_target_tau
                        )


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
