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


from isaacgym import gymapi, gymtorch  # type: ignore[misc]
import torch

from phys_anim.envs.mimic.isaacgym import MimicHumanoid

from hydra.utils import instantiate

from phys_anim.agents.models.actor import ActorFixedSigma

from pathlib import Path

class MimicHumanoidDelta(MimicHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)
        self.delta_action: ActorFixedSigma = instantiate(
            self.config.deltaaction, num_in=self.num_obs, num_act=self.num_act
        )
        self.delta_action.to(self.device)
        self.delta_action.eval()

        checkpoint = Path(self.config.delta_checkpoint).resolve()
        print(f"Loading delta from checkpoint: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=self.device)

        self.delta_action.load_state_dict(state_dict['actor'])

        self.orig_action = torch.zeros_like(self.initial_dof_pos, device=self.device, dtype=torch.float32)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        self.orig_action = actions

        delta = self.delta_action.eval_forward({'obs': self.obs_buf,
                                   'mimic_target_poses': self.mimic_target_poses,
                                   'mjc_action': actions})['mus']

        self.actions = actions + delta

    def post_physics_step(self):
        super().post_physics_step()
        self.actions = self.orig_action