


from typing import TYPE_CHECKING, Dict, Optional
import torch
from phys_anim.envs.mimic.isaacgym import MimicHumanoid

from isaac_utils import rotations, torch_utils

from torch import Tensor

from phys_anim.envs.mimic.mimic_utils import (
    build_max_coords_target_poses,
    build_max_coords_target_poses_future_rel,
    build_max_coords_object_target_poses,
    build_max_coords_object_target_poses_future_rel,
    dof_to_local,
    exp_tracking_reward,
)

from phys_anim.envs.humanoid.humanoid_utils import quat_diff_norm

class DeltaAction(MimicHumanoid):
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

        self.delta_action = torch.zeros_like(self.initial_dof_pos, device=self.device, dtype=torch.float32)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        motion_state_action = self.motion_lib.get_motion_state(0, self.motion_times)

        mjc_action = motion_state_action.action

        self.delta_action = actions
        self.actions = actions + mjc_action


    def post_physics_step(self):
        super().post_physics_step()
        self.actions = self.delta_action

    def reset(self, env_ids=None):
        """
        return:: new obs
        """
        super().reset(env_ids)
        return self.obs_buf
