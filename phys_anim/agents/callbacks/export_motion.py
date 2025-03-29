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
from phys_anim.envs.humanoid.common import BaseHumanoid

import torch
from pathlib import Path
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import os.path as osp
import shutil

import numpy as np
import yaml
import pickle
from scipy.spatial.transform import Rotation as sRot
import h5py


class ExportMotion(RL_EvalCallback):
    training_loop: PPO
    env: BaseHumanoid

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        self.record_dir = Path(config.record_dir)
        self.record_dir.mkdir(exist_ok=True, parents=True)
        self.num_rec_steps = 0
        self.num_steps = training_loop.num_steps
        self.file = h5py.File(self.record_dir / 'dataset.hdf5', 'w')
        self.num_envs = 0

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        env: BaseHumanoid = self.training_loop.env
        self.env = env
        self.num_envs = self.env.config.env_num_to_export
        length = self.training_loop.num_steps * self.training_loop.config.max_epochs
        self.file.create_dataset('obs',(length, self.num_envs,self.env.num_obs))
        self.file.create_dataset('mimic_target_poses',(length, self.num_envs, self.env.config.mimic_target_pose.num_obs_per_target_pose
                                                 * self.env.config.mimic_target_pose.num_future_steps))
        self.file.create_dataset('actions', (length, self.num_envs, self.env.num_act))
        self.file.create_dataset('rewards', (length, self.num_envs))
        self.file.create_dataset('dones', (length, self.num_envs))

    def on_pre_eval_env_step(self, actor_state):
        actor_state["actions"] = actor_state["mus"]
        return actor_state

    def on_post_train_play_steps(self, actor_state):
        start = self.num_rec_steps
        end = self.num_rec_steps + self.num_steps

        self.file['obs'][start:end] = actor_state["obs"][0: self.num_envs].cpu().numpy()
        self.file['mimic_target_poses'][start:end] = actor_state["mimic_target_poses"][0: self.num_envs].cpu().numpy()
        self.file['actions'][start:end] = actor_state["actions"][0: self.num_envs].cpu().numpy()
        self.file['rewards'][start:end] = actor_state["rewards"][0: self.num_envs].cpu().numpy()
        self.file['dones'][start:end] = actor_state["dones"][0: self.num_envs].cpu().numpy()

        self.num_rec_steps += self.num_steps

    def on_post_evaluate_policy(self):
        #self.write_recordings()
        self.file.close()
        pass

    def write_recordings(self):
        fps = np.round(1.0 / self.env.dt)
        for idx in range(self.env.config.env_num_to_export):
            trajectory_data = self.env.motion_recording

            save_dir = self.record_dir / f"{(idx):03d}"#+ self.config.index_offset
            save_dir.mkdir(exist_ok=True, parents=True)

            motion_name = self.env.motion_lib.state.motion_files[0].rsplit('/')[-1].split('.')[0]

            if self.config.store_poselib:
                skeleton_tree = self.env.motion_lib.state.motions[0].skeleton_tree

                curr_root_pos = torch.stack(
                    [root_pos[idx] for root_pos in trajectory_data["root_pos"]]
                )
                curr_body_rot = torch.stack(
                    [global_rot[idx] for global_rot in trajectory_data["global_rot"]]
                )

                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree, curr_body_rot, curr_root_pos, is_local=False
                )
                sk_motion = SkeletonMotion.from_skeleton_state(sk_state, fps=fps)

                sk_motion.to_file(str(save_dir / f"{motion_name}.npy"))

                #if "target_poses" in trajectory_data:
                #    target_poses = torch.tensor(
                #        np.stack(
                #            [
                #                target_pose[idx]
                #                for target_pose in trajectory_data["target_poses"]
                #            ]
                #        )
                #    )
                #    np.save(
                #        str(save_dir / f"target_poses_{idx}.npy"),
                #        target_poses.cpu().numpy(),
                #    )

                if "actions" in trajectory_data:
                    actions = torch.stack(
                        [actions[idx] for actions in trajectory_data["actions"]]
                    )
                    np.save(
                        str(save_dir / f"{motion_name}_actions.npy"),
                        actions.cpu().numpy()
                    )

                if "reset" in trajectory_data:
                    reset = torch.stack(
                        [resets[idx] for resets in trajectory_data["reset"]]
                    )

                    np.save(
                        str(save_dir / f"{motion_name}_reset.npy"),
                        reset.cpu().numpy()
                    )

                if hasattr(self.env, "object_ids") and self.env.object_ids[idx] >= 0:
                    object_id = self.env.object_ids[idx].item()
                    object_category, object_name = self.env.spawned_object_names[
                        object_id
                    ].split("_")
                    object_offset = self.env.object_offsets[object_category]
                    object_pos = self.env.scene_position[object_id].clone()
                    object_pos[0] += object_offset[0]
                    object_pos[1] += object_offset[1]

                    object_bbs = self.env.object_id_to_object_bounding_box[
                        object_id
                    ].clone()

                    # Add the height offset for the bounding box to match in global coords
                    object_center_xy = self.env.object_root_states[object_id, :2].view(
                        1, 2
                    )
                    terrain_height_below_object = self.env.get_ground_heights(
                        object_center_xy
                    ).view(1)
                    object_bbs[:, -1] += terrain_height_below_object

                    object_info = {
                        "object_pos": [
                            object_pos[0].item(),
                            object_pos[1].item(),
                            object_pos[2].item(),
                        ],
                        "object_name": object_name,
                        "object_bbox": object_bbs.cpu().tolist(),
                    }
                    with open(str(save_dir / f"object_info_{idx}.yaml"), "w") as file:
                        yaml.dump(object_info, file)
                    category_root = osp.join(
                        self.env.config.object_asset_root, object_category
                    )
                    # copy urdf and obj files to new dir, using copy functions
                    shutil.copyfile(
                        str(osp.join(category_root, f"{object_name}.urdf")),
                        str(save_dir / f"{object_name}.urdf"),
                    )
                    shutil.copyfile(
                        str(osp.join(category_root, f"{object_name}.obj")),
                        str(save_dir / f"{object_name}.obj"),
                    )

            else:
                for key in self.env.motion_recording.keys():
                    values = torch.stack(
                        [values[idx] for values in self.env.motion_recording[key]]
                    )
                    np.save(
                        str(save_dir / f"{motion_name}_{key}.npy"),
                        values.cpu().numpy()
                    )

        for key in self.env.motion_recording.keys():
            self.env.motion_recording[key] = []
