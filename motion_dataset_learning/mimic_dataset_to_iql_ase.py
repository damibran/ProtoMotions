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

'''
demo_dataset_files_paths = [
    'output/recordings/mimic_eval/RL_Avatar_Atk_2xCombo01_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_Atk_Kick_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_Atk_ShieldSwipe01_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_Counter_Atk03_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_Idle_Ready_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_RunBackward_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_RunForward_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_RunLeft_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_RunRight_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_TurnLeft90_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_TurnLeft180_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_TurnRight90_Motion/dataset.hdf5',
    'output/recordings/mimic_eval/RL_Avatar_TurnRight180_Motion/dataset.hdf5',
]
'''

demo_dataset_files_paths = [
    'output/recordings/mimic_train/RL_Avatar_Atk_2xCombo01_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_Atk_Kick_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_Atk_ShieldSwipe01_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_Counter_Atk03_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_Idle_Ready_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_RunBackward_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_RunForward_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_RunLeft_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_RunRight_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_TurnLeft90_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_TurnLeft180_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_TurnRight90_Motion/dataset.hdf5',
    'output/recordings/mimic_train/RL_Avatar_TurnRight180_Motion/dataset.hdf5',
]

new_dataset_files_paths = [path.replace('mimic_train', 'mimic_train_iql') for path in demo_dataset_files_paths]

demo_dataset_files = []
for path in demo_dataset_files_paths:
    demo_dataset_files.append(h5py.File(path, "r"))

new_demo_dataset_files = []
for idx, path in enumerate(new_dataset_files_paths):
    full_path = os.getcwd() + '/'+path
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    new_demo_dataset_files.append(h5py.File(full_path, "w"))
    for key in demo_dataset_files[idx]:
        demo_dataset_files[idx].copy(key, new_demo_dataset_files[idx])
    num_steps = new_demo_dataset_files[idx]['dones'].shape[0]
    num_envs = new_demo_dataset_files[idx]['dones'].shape[1]
    num_obs = 206 # env.config.discriminator_obs_size_per_step
    new_demo_dataset_files[idx].create_dataset("disc_obs",(num_steps, num_envs, num_obs))

skeleton_tree = SkeletonTree.from_mjcf('phys_anim/data/assets/mjcf/amp_humanoid_sword_shield.xml')

device = 'cpu'

dfs_dof_body_ids = [ 1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16 ] #all_config.robot.dfs_dof_body_ids
dfs_dof_names = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z',
                 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_elbow',
                 'right_hand_x', 'right_hand_y', 'right_hand_z', 'left_shoulder_x', 'left_shoulder_y',
                 'left_shoulder_z', 'left_elbow', 'right_hip_x', 'right_hip_y', 'right_hip_z',
                 'right_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_hip_x',
                 'left_hip_y', 'left_hip_z', 'left_knee', 'left_ankle_x', 'left_ankle_y',
                 'left_ankle_z'] #all_config.robot.dfs_dof_names
num_act = 31

dof_offsets = []
previous_dof_name = "null"
for dof_offset, dof_name in enumerate(dfs_dof_names):
    if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
        previous_dof_name = dof_name[:-2]
        dof_offsets.append(dof_offset)
dof_offsets.append(len(dfs_dof_names))

key_bodies = [
    "sword",
    "shield",
]

dfs_body_names = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'sword',
                 'left_upper_arm', 'left_lower_arm', 'shield', 'left_hand', 'right_thigh', 'right_shin', 'right_foot',
                 'left_thigh', 'left_shin', 'left_foot']

key_body_ids = torch.tensor(
            [
                dfs_body_names.index(key_body_name)
                for key_body_name in key_bodies
            ],
            dtype=torch.long,
        )

for idx, file in enumerate(demo_dataset_files):
    for env_id in range(new_demo_dataset_files[idx]['dones'].shape[1]):
        global_rot = torch.from_numpy(file['global_rot'][:, env_id, ...])
        root_pos = torch.from_numpy(file['root_pos'][:, env_id, ...])
        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            global_rot,
            root_pos,
            is_local=False
        )
        sk_motion = SkeletonMotion.from_skeleton_state(sk_state, 30)
        root_pos = sk_motion.root_translation.to(device)
        root_rot = sk_motion.global_root_rotation.to(device)
        root_vel = sk_motion.global_root_velocity.to(device)
        root_ang_vel = sk_motion.global_root_angular_velocity.to(device)
        dof_pos = _local_rotation_to_dof(dof_body_ids=dfs_dof_body_ids,
                                         dof_offsets=dof_offsets,
                                         num_dof=num_act,
                                         device=device,
                                         local_rot=sk_motion.local_rotation,
                                         joint_3d_format='exp_map', ).to(device)
        dof_vel = torch.from_numpy(file['dof_vel'][:, env_id, ...]).to(device) # for train [:, env_id, ...]
        key_body_pos = sk_motion.global_translation[:, key_body_ids].to(device)
        actions = torch.from_numpy(file['actions'][:, env_id, ...]).to(device)
        disc_obs = build_disc_action_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_body_pos,
            torch.zeros(1, device=device),
            actions,
            True,
            True,
            78,
            dof_offsets,
            False,
            True,
        )
        new_demo_dataset_files[idx]['disc_obs'][:, env_id] = disc_obs