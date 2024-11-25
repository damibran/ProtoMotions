from typing import List, Any

import numpy as np
import torch
from sympy.physics.units import action
from torch import Tensor
from easydict import EasyDict
from dataclasses import dataclass

from phys_anim.utils.motion_lib import MotionLib
from isaac_utils import rotations, torch_utils
from poselib.core.rotation3d import quat_angle_axis, quat_inverse, quat_mul_norm
from envs.humanoid.humanoid_utils import joint_dof_pos_to_vel

@dataclass
class MotionStateAction:
    root_pos: Tensor
    root_rot: Tensor
    dof_pos: Tensor
    root_vel: Tensor
    root_ang_vel: Tensor
    dof_vel: Tensor
    key_body_pos: Tensor
    rb_pos: Tensor
    rb_rot: Tensor
    local_rot: Tensor
    rb_vel: Tensor
    rb_ang_vel: Tensor
    action: Tensor

class StateActionLib(MotionLib):
    def __init__(
        self,
        motion_file,
        dof_body_ids,
        dof_offsets,
        key_body_ids,
        device="cpu",
        ref_height_adjust: float = 0,
        target_frame_rate: int = 30,
        w_last: bool = True,
        create_text_embeddings: bool = False,
        spawned_scene_ids: List[str] = None,
        fix_motion_heights: bool = False,
        skeleton_tree: Any = None,
    ):
        super().__init__(
            motion_file=motion_file,
            dof_body_ids=dof_body_ids,
            dof_offsets=dof_offsets,
            key_body_ids=key_body_ids,
            device=device,
            ref_height_adjust=ref_height_adjust,
            target_frame_rate=target_frame_rate,
            w_last=w_last,
            create_text_embeddings=create_text_embeddings,
            spawned_scene_ids=spawned_scene_ids,
            fix_motion_heights=fix_motion_heights,
            skeleton_tree=skeleton_tree,
        )

        files_actions = []
        motion_dir = '/'.join(motion_file.rsplit("/")[:-1])
        for m_file in self.state.motion_files:
            name = m_file.rsplit("/")[-1].split(".")[0]
            actions_file = motion_dir + "/" + name + "_actions.npy"
            actions = torch.Tensor(np.load(actions_file)).to(self.device)
            actions = actions.squeeze_(1)
            files_actions.append(actions)

        self.register_buffer(
            "actions",
            torch.cat([act for act in files_actions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )

        pass

    def get_motion_state(self, sub_motion_ids, motion_times, joint_3d_format="exp_map")\
            -> MotionStateAction:

        motion_state = super().get_motion_state(sub_motion_ids, motion_times)

        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]
        motion_len = self.state.motion_lengths[motion_ids]
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        blend = blend.unsqueeze(-1)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        action0 = self.actions[f0l]
        action1 = self.actions[f1l]

        action = (1.0 - blend) * action0 + blend * action1

        motion_state_action = MotionStateAction(
            root_pos=motion_state.root_pos,
            root_rot=motion_state.root_rot,
            root_vel=motion_state.root_vel,
            root_ang_vel=motion_state.root_ang_vel,
            key_body_pos=motion_state.key_body_pos,
            dof_pos=motion_state.dof_pos,
            dof_vel=motion_state.dof_vel,
            local_rot=motion_state.local_rot,
            rb_pos=motion_state.rb_pos,
            rb_rot=motion_state.rb_rot,
            rb_vel=motion_state.rb_vel,
            rb_ang_vel=motion_state.rb_ang_vel,
            action=action
        )

        return motion_state_action