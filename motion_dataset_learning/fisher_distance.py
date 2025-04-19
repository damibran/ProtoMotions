import os

from omegaconf import OmegaConf
from phys_anim.agents.models.infomax import JointDiscWithMutualInformationEncMLP
from phys_anim.utils.StateActionLib import StateActionLib
from poselib.skeleton.skeleton3d import SkeletonMotion
from phys_anim.envs.humanoid.humanoid_utils import build_disc_action_observations
import torch


config = OmegaConf.load('motion_dataset_learning/config/fisher_distance.yaml')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_disc_hist_step = config.discriminator.config.discriminator_obs_historical_steps

encoder = JointDiscWithMutualInformationEncMLP(config.discriminator.config,
                                               num_in = config.discriminator_obs_size_per_step * num_disc_hist_step)


encoder.load_state_dict(torch.load(config.discriminator.config.checkpoint)["discriminator"])
encoder.to(device)

dof_body_ids = config.robot.dfs_dof_body_ids
dof_offsets = []
previous_dof_name = "null"
for dof_offset, dof_name in enumerate(config.robot.dfs_dof_names):
    if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
        previous_dof_name = dof_name[:-2]
        dof_offsets.append(dof_offset)
dof_offsets.append(len(config.robot.dfs_dof_names))
dof_obs_size = config.robot.dof_obs_size
num_act = config.robot.number_of_actions

key_body_ids = torch.tensor(
                [
                    config.robot.dfs_body_names.index(key_body_name)
                    for key_body_name in config.robot.key_bodies
                ],
                dtype=torch.long,
            )

motion_lib = StateActionLib(
            motion_file=config.motion_file,
            dof_body_ids=dof_body_ids,
            dof_offsets=dof_offsets,
            key_body_ids=key_body_ids,
            device=device,
        )

def get_motion_windows_count(motion_id):
    return motion_lib.state.motion_num_frames[motion_id] - num_disc_hist_step

def get_motions_sum_encoder_distance(motion_id, other_motion_id):
    # Get total number of windows
    motion_window_count = get_motion_windows_count(motion_id)
    other_motion_window_count = get_motion_windows_count(other_motion_id)

    # Build all observation tensors for motion_id
    obs_list_1 = []
    for i in range(motion_window_count - 1):
        state = motion_lib.get_window_state_for_disc(i, motion_id, num_disc_hist_step)
        obs = build_disc_action_observations(
            state.root_pos,
            state.root_rot,
            state.root_vel,
            state.root_ang_vel,
            state.dof_pos,
            state.dof_vel,
            state.key_body_pos,
            torch.zeros(1, device=device),
            state.action,
            True, True, config.robot.dof_obs_size, dof_offsets, False, True
        )
        obs_list_1.append(obs.flatten())

    # Build all observation tensors for other_motion_id
    obs_list_2 = []
    for i in range(other_motion_window_count - 1):
        state = motion_lib.get_window_state_for_disc(i, other_motion_id, num_disc_hist_step)
    obs = build_disc_action_observations(
        state.root_pos,
        state.root_rot,
        state.root_vel,
        state.root_ang_vel,
        state.dof_pos,
        state.dof_vel,
        state.key_body_pos,
        torch.zeros(1, device=device),
        state.action,
        True, True, config.robot.dof_obs_size, dof_offsets, False, True
    )
    obs_list_2.append(obs.flatten())

    # Stack and encode all observations
    obs_tensor_1 = torch.stack(obs_list_1)  # Shape: [N1, D]
    obs_tensor_2 = torch.stack(obs_list_2)  # Shape: [N2, D]

    z1 = encoder({'obs': obs_tensor_1}, return_enc=True)  # Shape: [N1, E]
    z2 = encoder({'obs': obs_tensor_2}, return_enc=True)  # Shape: [N2, E]

    # Compute all pairwise distances
    # z1: [N1, E], z2: [N2, E]
    # Compute pairwise Euclidean distance matrix: [N1, N2]
    diff = z1[:, None, :] - z2[None, :, :]  # [N1, N2, E]
    dists = torch.norm(diff, dim=2)         # [N1, N2]
    total_dist = dists.sum().item()

    return total_dist

all_coeffs = []
with open('motion_dataset_learning/fisher_distance_result/fisher_distance_result.txt','w') as out_file:
    for motion_id in range(len(motion_lib.state.motion_files)):
        out_file.write(str(motion_lib.state.motion_files[motion_id]) + '\n')
        print(motion_lib.state.motion_files[motion_id])
        self_sum = get_motions_sum_encoder_distance(motion_id, motion_id) / (pow(get_motion_windows_count(motion_id),2))
        all_sum_distance = 0
        all_sum_window_count = 0
        for other_motion_id in range(len(motion_lib.state.motion_files)):
            all_sum_distance += get_motions_sum_encoder_distance(motion_id ,other_motion_id)
            all_sum_window_count += get_motion_windows_count(other_motion_id)
        all_sum_distance = all_sum_distance / (get_motion_windows_count(motion_id) * all_sum_window_count)
        coeff = self_sum / all_sum_distance
        out_file.write(str(coeff) + '\n')
        print(coeff)
        all_coeffs.append(coeff)
    avg = sum(all_coeffs) / len(all_coeffs)
    out_file.write('\n\nAVG: ' + str(avg) + '\n')
    print(avg)
    pass