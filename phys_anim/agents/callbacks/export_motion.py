# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from phys_anim.agents.callbacks.base_callback import RL_EvalCallback
from phys_anim.agents.ppo import PPO
from phys_anim.envs.mimic.mujoco import MjcMimic, rot_conv_mjc_to_isaac

import torch
from pathlib import Path
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import os.path as osp
import shutil

import numpy as np
import yaml
import pickle
from scipy.spatial.transform import Rotation as sRot

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree


class ExportMotion(RL_EvalCallback):
    training_loop: PPO
    env: MjcMimic

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        self.record_dir = Path(config.record_dir)
        self.record_dir.mkdir(exist_ok=True, parents=True)
        self.trajectory_data=[]
        self.record_period = 32 * 100
        self.num_steps = 0

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        env: MjcMimic = self.training_loop.env
        self.env = env
        self.num_envs = self.env.config.env_num_to_export
        for env_id in range(self.num_envs):
            self.trajectory_data.append(
                {
                    "root_pos":[],
                    "global_rot":[],
                    'actions':[]
                }
            )

    def on_pre_train_env_step(self, actor_state):
        self.on_pre_env_step(actor_state)

    def on_pre_eval_env_step(self, actor_state):
        actor_state["actions"] = actor_state["sampled_actions"]
        return actor_state

    def on_pre_env_step(self, actor_state):
        self.num_steps += 1
        for env_id in range(self.num_envs):
            self.trajectory_data[env_id]['actions'].append(actor_state["actions"][env_id].cpu().numpy())
            self.trajectory_data[env_id]['root_pos'].append(self.env.mjc_datas[env_id].qpos[0:3].copy())
            self.trajectory_data[env_id]['global_rot'].append(self.env.mjc_datas[env_id].xquat[1:].copy())
        if self.num_steps % self.record_period == 0:
            self.write_recordings()

    def on_post_evaluate_policy(self):
        self.write_recordings()

    def write_recordings(self):
        fps = np.round(1.0 / self.env.dt)
        for idx in range(self.env.config.env_num_to_export):
            trajectory_data = self.trajectory_data[idx]

            save_dir = self.record_dir / f"{(idx):03d}"#+ self.config.index_offset
            save_dir.mkdir(exist_ok=True, parents=True)

            curr_root_pos = torch.stack(
                [torch.from_numpy(root_pos) for root_pos in trajectory_data["root_pos"]]
            )
            curr_body_rot = torch.stack(
                [torch.from_numpy(global_rot[:, rot_conv_mjc_to_isaac]) for global_rot in trajectory_data["global_rot"]]
            )

            skeleton_tree = SkeletonTree.from_mjcf('phys_anim/data/assets/mjcf/mjc_amp_humanoid_sword_shield.xml')

            sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree, curr_body_rot, curr_root_pos, is_local=False
            )
            sk_motion = SkeletonMotion.from_skeleton_state(sk_state, fps=30)

            motion_name = self.env.motion_lib.state.motion_files[0].split('/')[-1].split('.')[0]

            sk_motion.to_file(str(save_dir / f"{motion_name}.npy"))

            if "actions" in trajectory_data:
                actions = torch.stack(
                    [torch.from_numpy(actions) for actions in trajectory_data["actions"]]
                )
                np.save(
                    str(save_dir / f"{motion_name}_actions.npy"),
                    actions.cpu().numpy()
                )
