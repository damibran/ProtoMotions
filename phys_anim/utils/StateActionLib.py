from typing import List, Any

import numpy as np
import torch
from easydict import EasyDict

from phys_anim.utils.motion_lib import MotionLib
from isaac_utils import rotations, torch_utils
from poselib.core.rotation3d import quat_angle_axis, quat_inverse, quat_mul_norm
from envs.humanoid.humanoid_utils import joint_dof_pos_to_vel

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

        dir = '/'.join(motion_file.rsplit("/")[:-1])
        name = motion_file.rsplit("/")[-1].split(".")[0]

        actions_file = dir + "/" + name + "_actions.npy"

        self.actions = torch.Tensor(np.load(actions_file))

        self.actions = self.actions.squeeze_(1)

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

        self.register_buffer(
            "dof_pos",
            self.actions.to(
                device=device, dtype=torch.float32
            ),
            persistent=False,
        )

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        raise NotImplemented()

    def _compute_motion_dof_vels(self, motion):
        dt = 1.0 / motion.fps
        return joint_dof_pos_to_vel(self.actions, self.dof_offsets, dt, self.w_last)