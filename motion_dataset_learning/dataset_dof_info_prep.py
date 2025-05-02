import os

import torch

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
from poselib.core.rotation3d import quat_angle_axis, quat_inverse, quat_mul_norm
from isaac_utils import rotations, torch_utils
import h5py
from phys_anim.envs.humanoid.humanoid_utils import build_disc_action_observations, compute_humanoid_observations_max


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

def _local_rotation_to_dof_vel(dof_body_ids,dof_offsets, num_dof, device, local_rot0, local_rot1, dt):
    body_ids = dof_body_ids
    dof_offsets = dof_offsets

    dof_vel = torch.zeros([num_dof], device=device)

    diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
    local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
    local_vel = local_vel

    for j in range(len(body_ids)):
        body_id = body_ids[j]
        joint_offset = dof_offsets[j]
        joint_size = dof_offsets[j + 1] - joint_offset

        if joint_size == 3:
            joint_vel = local_vel[body_id]
            dof_vel[joint_offset : (joint_offset + joint_size)] = joint_vel

        elif joint_size == 1:
            assert joint_size == 1
            joint_vel = local_vel[body_id]
            dof_vel[joint_offset] = joint_vel[
                1
            ]  # assume joint is always along y axis

        else:
            print("Unsupported joint type")
            assert False

    return dof_vel


def _compute_motion_dof_vels(dof_body_ids,dof_offsets, num_dof, device,motion: SkeletonMotion):
    num_frames = motion.global_translation.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = _local_rotation_to_dof_vel(dof_body_ids,dof_offsets, num_dof, device,local_rot0, local_rot1, dt)
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0)

    return dof_vels

skeleton_tree = SkeletonTree.from_mjcf('phys_anim/data/assets/mjcf/amp_humanoid_sword_shield.xml')

demo_dataset_files_paths = [
    'output/recordings/mimic_train_iql/RL_Avatar_Atk_2xCombo01_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_Atk_Kick_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_Atk_ShieldSwipe01_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_Counter_Atk03_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_Idle_Ready_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_RunBackward_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_RunForward_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_RunLeft_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_RunRight_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_TurnLeft90_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_TurnLeft180_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_TurnRight90_Motion/dataset.hdf5',
    'output/recordings/mimic_train_iql/RL_Avatar_TurnRight180_Motion/dataset.hdf5',
]

num_act = 31
hum_num_obs = 253

demo_dataset_files = []
for path in demo_dataset_files_paths:
    file = h5py.File(path, "r+")
    demo_dataset_files.append(file)
    num_steps = file['dones'].shape[0]
    num_envs = file['dones'].shape[1]
    file.create_dataset("dof_pos", (num_steps, num_envs, num_act))
    file.create_dataset("human_obs", (num_steps, num_envs, hum_num_obs))

dfs_dof_body_ids = [ 1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16 ] #all_config.robot.dfs_dof_body_ids
dfs_dof_names = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z',
                 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_elbow',
                 'right_hand_x', 'right_hand_y', 'right_hand_z', 'left_shoulder_x', 'left_shoulder_y',
                 'left_shoulder_z', 'left_elbow', 'right_hip_x', 'right_hip_y', 'right_hip_z',
                 'right_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_hip_x',
                 'left_hip_y', 'left_hip_z', 'left_knee', 'left_ankle_x', 'left_ankle_y',
                 'left_ankle_z'] #all_config.robot.dfs_dof_names

dof_offsets = []
previous_dof_name = "null"
for dof_offset, dof_name in enumerate(dfs_dof_names):
    if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
        previous_dof_name = dof_name[:-2]
        dof_offsets.append(dof_offset)
dof_offsets.append(len(dfs_dof_names))

device = 'cpu'

for file in demo_dataset_files:
    for env_id in range(file['dones'].shape[1]):
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
        human_obs = compute_humanoid_observations_max(
            sk_motion.global_translation.to(device),
            sk_motion.global_rotation.to(device),
            sk_motion.global_velocity.to(device),
            sk_motion.global_angular_velocity.to(device),
            torch.zeros(1, device=device),
            True,
            True,
            True,
        )
        file['dof_pos'][:, env_id] = dof_pos
        file['human_obs'][:, env_id] = human_obs