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

import torch
from torch import Tensor

from phys_anim.envs.amp.common import BaseDisc
from phys_anim.envs.humanoid.isaacgym import Humanoid
from phys_anim.envs.humanoid.humanoid_utils import build_disc_state_action_observations, build_disc_action_observations


class DiscHumanoid(BaseDisc, Humanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

class DiscActionHumanoid(DiscHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)
        self.actions = torch.zeros_like(self.dof_pos,device=self.device, dtype=torch.float32)

    def reset_disc_hist_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self.discriminator_obs_historical_steps - 1]
        )
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (
            torch.arange(
                0, self.discriminator_obs_historical_steps - 1, device=self.device
            )
            + 1
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)

        disc_obs_demo = build_disc_action_observations(
            ref_state.action,
            self.dof_obs_size,
            self.get_dof_offsets(),
            False,
            self.w_last,
        )
        if self.discriminator_obs_historical_steps >= 2:
            self.disc_hist_buf.set_hist(
                disc_obs_demo.view(
                    len(env_ids), self.discriminator_obs_historical_steps - 1, -1
                ).permute(1, 0, 2),
                env_ids,
            )

    def build_disc_obs_demo(self, motion_ids: Tensor, motion_times0: Tensor):
        dt = self.dt

        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self.discriminator_obs_historical_steps]
        )
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(
            0, self.discriminator_obs_historical_steps, device=self.device
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        # motion ids above are "sub_motions" so first we map to motion file itself and then extract the length.
        lengths = self.motion_lib.state.motion_lengths[
            self.motion_lib.state.sub_motion_to_motion[motion_ids]
        ]

        assert torch.all(motion_times >= 0)
        assert torch.all(motion_times <= lengths)

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)

        disc_obs_demo = build_disc_action_observations(
            ref_state.action,
            self.dof_obs_size,
            self.get_dof_offsets(),
            False,
            self.w_last,
        )
        return disc_obs_demo

    def compute_disc_observations(self, env_ids=None):
        current_state = self.get_bodies_state()

        dof_pos, dof_vel = self.get_dof_state()
        key_body_pos = current_state.body_pos[:, self.key_body_ids, :]

        if env_ids is None:
            disc_obs = build_disc_action_observations(
                self.actions,
                self.dof_obs_size,
                self.get_dof_offsets(),
                False,
                self.w_last,
            )
            self.disc_hist_buf.set_curr(disc_obs)
        else:
            disc_obs = build_disc_action_observations(
                self.actions[env_ids],
                self.dof_obs_size,
                self.get_dof_offsets(),
                False,
                self.w_last,
            )
            self.disc_hist_buf.set_curr(disc_obs, env_ids)