import os.path
import random
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch import nn as nn
from torch import Tensor
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


# def list_roll(inlist, n):
#    for i in range(n):
#        inlist.append(inlist.pop(0))

class BC:

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

        print("Demo Dataset Processing START")
        motion_libs = []
        for m_file in self.config.motion_files:
            motion_libs.append(StateActionLib(
                motion_file=m_file,
                dof_body_ids=self.dof_body_ids,
                dof_offsets=self.dof_offsets,
                key_body_ids=self.key_body_ids,
                device=self.device,
            ))

        print("Demo Dataset Libs Loaded")
        self.demo_subsets = []
        state = None
        for lib in motion_libs:
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
                    )
                }

                for attr in lib_set[motion_id].keys():
                    if attr == "Actions":
                        # Shift tensor elements to the left and trim size
                        lib_set[motion_id][attr] = torch.roll(lib_set[motion_id][attr], shifts=-1, dims=0)[:-1]
                    else:
                        # Remove the last entry for other tensors
                        lib_set[motion_id][attr] = lib_set[motion_id][attr][:-1]
            self.demo_subsets.append(lib_set)

        print("Demo Dataset Creating")
        self.demo_dataset = {
            "root_pos": torch.zeros(motion_libs[0].actions.shape[0], state.root_pos.shape[1], device=self.device),
            "root_rot": torch.zeros(motion_libs[0].actions.shape[0], state.root_rot.shape[1], device=self.device),
            "root_vel": torch.zeros(motion_libs[0].actions.shape[0], state.root_vel.shape[1], device=self.device),
            "root_ang_vel": torch.zeros(motion_libs[0].actions.shape[0], state.root_ang_vel.shape[1],
                                        device=self.device),
            "dof_pos": torch.zeros(motion_libs[0].actions.shape[0], state.dof_pos.shape[1], device=self.device),
            "dof_vel": torch.zeros(motion_libs[0].actions.shape[0], state.dof_vel.shape[1], device=self.device),
            "key_body_pos": torch.zeros(motion_libs[0].actions.shape[0], state.key_body_pos.shape[1],
                                        state.key_body_pos.shape[2], device=self.device),
            "Actions": torch.zeros(motion_libs[0].actions.shape[0], self.num_act, device=self.device),
            "HumanoidObservations": torch.zeros(motion_libs[0].actions.shape[0], self.num_obs, device=self.device)
        }

        self.demo_dataset_len = motion_libs[0].actions.shape[0]

        self.update_steps_per_stage = 1

        pass

    def fill_dataset(self):
        subset_ind = random.randint(0, len(self.demo_subsets) - 1)
        motion_ids = list(self.demo_subsets[subset_ind].keys())
        ds_strt_ind = 0
        while len(motion_ids) > 0:
            motion_id = random.choice(motion_ids)
            motion_ids.remove(motion_id)
            state = self.demo_subsets[subset_ind][motion_id]
            state_len = state["root_pos"].shape[0]
            self.demo_dataset["root_pos"][ds_strt_ind:ds_strt_ind + state_len] = state["root_pos"]
            self.demo_dataset["root_rot"][ds_strt_ind:ds_strt_ind + state_len] = state["root_rot"]
            self.demo_dataset["root_vel"][ds_strt_ind:ds_strt_ind + state_len] = state["root_vel"]
            self.demo_dataset["root_ang_vel"][ds_strt_ind:ds_strt_ind + state_len] = state["root_ang_vel"]
            self.demo_dataset["dof_pos"][ds_strt_ind:ds_strt_ind + state_len] = state["dof_pos"]
            self.demo_dataset["dof_vel"][ds_strt_ind:ds_strt_ind + state_len] = state["dof_vel"]
            self.demo_dataset["key_body_pos"][ds_strt_ind:ds_strt_ind + state_len] = state["key_body_pos"]
            self.demo_dataset["HumanoidObservations"][ds_strt_ind:ds_strt_ind + state_len] = state[
                "HumanoidObservations"]
            self.demo_dataset["Actions"][ds_strt_ind:ds_strt_ind + state_len] = state["Actions"]
            ds_strt_ind += state_len

    def dataset_roll(self):

        self.demo_dataset["root_pos"] = torch.roll(self.demo_dataset["root_pos"], shifts=-self.config.batch_size)
        self.demo_dataset["root_rot"] = torch.roll(self.demo_dataset["root_rot"], shifts=-self.config.batch_size)
        self.demo_dataset["root_vel"] = torch.roll(self.demo_dataset["root_vel"], shifts=-self.config.batch_size)
        self.demo_dataset["root_ang_vel"] = torch.roll(self.demo_dataset["root_ang_vel"],
                                                       shifts=-self.config.batch_size)
        self.demo_dataset["dof_pos"] = torch.roll(self.demo_dataset["dof_pos"], shifts=-self.config.batch_size)
        self.demo_dataset["dof_vel"] = torch.roll(self.demo_dataset["dof_vel"], shifts=-self.config.batch_size)
        self.demo_dataset["key_body_pos"] = torch.roll(self.demo_dataset["key_body_pos"],
                                                       shifts=-self.config.batch_size)
        self.demo_dataset["HumanoidObservations"] = torch.roll(self.demo_dataset["HumanoidObservations"],
                                                               shifts=-self.config.batch_size)
        self.demo_dataset["Actions"] = torch.roll(self.demo_dataset["Actions"], shifts=-self.config.batch_size)

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

        #state_dict = torch.load(Path.cwd() / "results/iql/lightning_logs/version_6/last.ckpt", map_location=self.device)
        #self.actor.load_state_dict(state_dict["actor"])
        #self.save(name="last_a.ckpt")

    def fit(self):

        for self.current_epoch in range(self.config.max_epochs):
            print(f"Epoch: {self.current_epoch}")
            batch_count = math.ceil(self.demo_dataset_len / self.config.batch_size)

            self.fill_dataset()

            print(f'Value Step')

            a_loss_tensor = torch.zeros(self.update_steps_per_stage * batch_count)
            print(f'Actor Step')
            for i in range(self.update_steps_per_stage):
                for batch_id in range(batch_count):
                    self.dataset_roll()

                    batch = {
                        "HumanoidObservations": self.demo_dataset["HumanoidObservations"][0:self.config.batch_size],
                        "Actions": self.demo_dataset["Actions"][0:self.config.batch_size]
                    }

                    self.actor.training = True
                    actor_out = self.actor.training_forward(
                        {"obs": batch["HumanoidObservations"], "actions": batch["Actions"]})

                    actor_loss = actor_out["neglogp"].mean()

                    a_loss_tensor[batch_id * self.update_steps_per_stage + i] = actor_loss.mean().detach()


                    self.actor_optimizer.zero_grad()
                    self.fabric.backward(actor_loss.mean())
                    self.actor_optimizer.step()

            self.log_dict.update({"actor_loss/actor_loss": a_loss_tensor.mean()})

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
            "actor_optimizer": self.actor_optimizer.state_dict(),
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
