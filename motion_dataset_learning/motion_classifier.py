import random

from omegaconf import OmegaConf
from phys_anim.agents.models.infomax import JointDiscWithMutualInformationEncMLP
from phys_anim.utils.StateActionLib import StateActionLib
from poselib.skeleton.skeleton3d import SkeletonMotion
from phys_anim.envs.humanoid.humanoid_utils import build_disc_action_observations
import torch
from phys_anim.agents.models.mlp import MLP_WithNorm
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

config = OmegaConf.load('motion_dataset_learning/config/classifier.yml')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter(log_dir="motion_dataset_learning/runs/classifier_training")

num_disc_hist_step = config.classifier.config.discriminator_obs_historical_steps

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
num_classes = len(motion_lib.state.motion_files)

# Create classifier model
classifier = MLP_WithNorm(
    config.classifier.config,
    num_in=config.discriminator_obs_size_per_step * num_disc_hist_step,
    num_out=num_classes,
).to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 1000
batch_size = 128
epoch = 0

for _ in tqdm(range(num_epochs), desc=f"Epoch {epoch}"):
    obs_batch = []
    label_batch = []
    classifier.train()

    for _ in range(batch_size):
        rand_motion = random.randint(0, num_classes - 1)
        rand_window_start = random.randint(
            0,
            motion_lib.state.motion_num_frames[rand_motion] - num_disc_hist_step - 2
        )
        ref_state = motion_lib.get_window_state_for_disc(
            rand_window_start,
            rand_motion,
            num_disc_hist_step
        )
        obs = build_disc_action_observations(
            ref_state.root_pos,
            ref_state.root_rot,
            ref_state.root_vel,
            ref_state.root_ang_vel,
            ref_state.dof_pos,
            ref_state.dof_vel,
            ref_state.key_body_pos,
            torch.zeros(1, device=device),
            ref_state.action,
            True,
            True,
            config.robot.dof_obs_size,
            dof_offsets,
            False,
            True,
        )
        obs_batch.append(obs.flatten())
        label_batch.append(rand_motion)

    obs_batch = torch.stack(obs_batch, dim=0)
    label_batch = torch.tensor(label_batch, dtype=torch.long, device=device)

    logits = classifier({'obs': obs_batch})
    loss = criterion(logits, label_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Avg Loss: {loss.mean().detach():.4f}")
    writer.add_scalar("Loss/train", loss.mean().detach(), epoch)
    epoch += 1

writer.close()