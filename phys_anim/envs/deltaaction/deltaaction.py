

import torch
from phys_anim.envs.mimic.isaacgym import MimicHumanoid


class DeltaAction(MimicHumanoid):
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

        self.delta_action = torch.zeros_like(self.initial_dof_pos, device=self.device, dtype=torch.float32)
        self.mjc_action = torch.zeros((self.num_envs, self.num_act), device=self.device, dtype=torch.float32)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        self.delta_action = actions
        self.actions = actions + self.mjc_action


    def post_physics_step(self):
        super().post_physics_step()
        self.actions = self.delta_action

    def step(self, actions):
        super().step(actions)

        motion_state_action = self.motion_lib.get_motion_state(0, self.motion_times)
        self.mjc_action = motion_state_action.action

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None):
        """
        return:: new obs
        """
        super().reset(env_ids)

        motion_state_action = self.motion_lib.get_motion_state(0, self.motion_times)
        self.mjc_action = motion_state_action.action

        return self.obs_buf
