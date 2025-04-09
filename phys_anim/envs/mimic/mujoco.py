import torch
from phys_anim.envs.base_interface.common import BaseInterface
from hydra.utils import instantiate
from phys_anim.utils.motion_lib import MotionLib
import mujoco
import phys_anim.envs.humanoid.humanoid_utils as humanoid_utils
import phys_anim.envs.mimic.mimic_utils as mimic_utils
import numpy as np
from isaac_utils import torch_utils, rotations
from torch import Tensor
from typing import Dict, List
import os
import threading

rot_conv_isaac_to_mjc = np.array([3, 0, 1, 2])
rot_conv_mjc_to_isaac = np.array([1,2,3,0])

class MjcMimic(BaseInterface):
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

        self.mjc_models = []
        self.mjc_datas = []

        for _ in range(self.config.num_envs):
            mjc_model = mujoco.MjModel.from_xml_path(
                os.getcwd() + '/phys_anim/data/assets/mjcf/mjc_amp_humanoid_sword_shield.xml')
            self.mjc_models.append(mjc_model)
            self.mjc_datas.append(mujoco.MjData(mjc_model))

        self.dt = self.mjc_models[0].opt.timestep

        min_val = 100.
        max_val = -100.
        # Iterate through all actuators
        for mjc_model in self.mjc_models:
            for i in range(mjc_model.nu):  # Number of actuators
                # Get the joint ID linked to this actuator
                joint_id = mjc_model.actuator_trnid[i, 0]  # First column refers to the joint

                if joint_id >= 0:  # Ensure it's a valid joint
                    # Copy joint properties to the actuator
                    mjc_model.actuator_gainprm[i, 0] = mjc_model.jnt_stiffness[joint_id]  # Stiffness
                    mjc_model.actuator_biasprm[i, 0] = mjc_model.dof_damping[joint_id]  # Damping

                    # Copy joint limits if needed
                    if mjc_model.jnt_limited[joint_id]:
                        mjc_model.actuator_ctrlrange[i, 0] = mjc_model.jnt_range[joint_id, 0]  # Lower limit
                        mjc_model.actuator_ctrlrange[i, 1] = mjc_model.jnt_range[joint_id, 1]  # Upper limit
                        if mjc_model.actuator_ctrlrange[i, 0] < min_val:
                            min_val = mjc_model.actuator_ctrlrange[i, 0]
                        if mjc_model.jnt_range[joint_id, 1] > max_val:
                            max_val = mjc_model.jnt_range[joint_id, 1]

        print(f'actuator range min: {min_val}, max: {max_val}')

        self.body_num = 17
        self.setup_character_props()

        self.key_body_ids = torch.tensor(
            [
                self.config.robot.dfs_body_names.index(key_body_name)
                for key_body_name in self.config.robot.key_bodies
            ],
            dtype=torch.long,
        )

        self.motion_lib: MotionLib = instantiate(
            self.config.motion_lib,
            dof_body_ids=self.dof_body_ids,
            dof_offsets=self.dof_offsets,
            key_body_ids=self.key_body_ids,
            device=self.device,
            skeleton_tree=None,
        )

        self.motion_times = torch.zeros(self.config.num_envs, device=self.device)

        self.dof_limits_lower = torch.from_numpy(self.mjc_models[0].actuator_ctrlrange[:, 0])
        self.dof_limits_upper = torch.from_numpy(self.mjc_models[0].actuator_ctrlrange[:, 1])

        self._pd_action_offset, self._pd_action_scale = (
            humanoid_utils.build_pd_action_offset_scale(
                self.dof_offsets,
                self.dof_limits_lower,
                self.dof_limits_upper,
                self.device,
            )
        )

        self.obs_buf = torch.zeros((self.config.num_envs, self.config.robot.self_obs_size), device=self.device)
        self.mimic_target_poses = torch.zeros((self.config.num_envs,
                                               self.config.mimic_target_pose.num_future_steps * self.config.mimic_target_pose.num_obs_per_target_pose),
                                              device=self.device)
        self.rew_buf = torch.zeros(self.config.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.config.num_envs, device=self.device)
        self.last_scaled_rewards: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.config.num_envs)]
        self.last_other_rewards: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.config.num_envs)]

        # Threading-related attributes
        self.threads = []

    def _thread_step(self, env_id, actions):
        pd_tar = self.action_to_pd_targets(actions[env_id])

        self.mjc_datas[env_id].ctrl = pd_tar.squeeze(0).cpu().detach().numpy()

        for i in range(10):
            mujoco.mj_step(self.mjc_models[env_id], self.mjc_datas[env_id])

        self.motion_times[env_id] += 1. / 30.
        self.obs_buf[env_id], self.mimic_target_poses[env_id] = self._compute_obs(env_id)
        self.rew_buf[env_id] = self._compute_reward(env_id)
        self.reset_buf[env_id] = self.motion_times[env_id] > self.motion_lib.get_motion_length(0) or \
                                 self.last_other_rewards[env_id][
                                     "max_joint_err"] > 0.25

    def step(self, actions: torch.Tensor):
        """
        return:: obs, rewards, dones, extras
        """
        self.threads = []
        # Create and start threads
        for env_id in range(self.config.num_envs):
            thread = threading.Thread(target=self._thread_step, args=(env_id, actions))
            self.threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in self.threads:
            thread.join()

        return self.obs_buf, self.rew_buf, self.reset_buf, {"to_log": {}, "terminate": self.reset_buf}

    def _thread_reset(self, env_id):
        mujoco.mj_resetData(self.mjc_models[env_id], self.mjc_datas[env_id])
        self.motion_times[env_id] = torch.zeros(1, device=self.device)
        self.prev_time = 0
        self._reset_humanoid(env_id)
        self.obs_buf[env_id], self.mimic_target_poses[env_id] = self._compute_obs(env_id)

    def reset(self, env_ids=None):
        """
        return:: new obs
        """
        if env_ids is None:
            env_ids = torch.arange(self.config.num_envs)

        self.threads = []

        # Create and start threads
        for env_id in env_ids:
            thread = threading.Thread(target=self._thread_reset, args=(env_id,))
            self.threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in self.threads:
            thread.join()

        return self.obs_buf

    def action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _reset_humanoid(self, env_id: int):
        ref_state = self.motion_lib.get_mimic_motion_state(
            0, self.motion_times[env_id].unsqueeze(0),
        )

        root_pos = ref_state.rb_pos[:,0,:].cpu().numpy()
        root_rot = ref_state.rb_rot[:,0,:].cpu().numpy()

        root_pos[:,2] += 0.2

        root_rot = root_rot[:, rot_conv_isaac_to_mjc]

        root_vel = ref_state.rb_vel[:,0,:].cpu().numpy()
        root_ang_vel = ref_state.rb_ang_vel[:,0,:].cpu().numpy()

        self.mjc_datas[env_id].qpos[0:3] = root_pos
        self.mjc_datas[env_id].qpos[3:7] = root_rot

        dof_pos = ref_state.dof_pos.cpu().numpy()
        joint_offset = 3 + 4
        self.mjc_datas[env_id].qpos[joint_offset:joint_offset + self.num_act] = dof_pos

        self.mjc_datas[env_id].qvel[0:3] = root_vel
        self.mjc_datas[env_id].qvel[3:6] = root_ang_vel
        self.mjc_datas[env_id].qvel[joint_offset - 1:joint_offset - 1 + self.num_act] = ref_state.dof_vel.cpu().numpy()

        pass

    def _compute_reward(self, env_id):

        ref_state = self.motion_lib.get_mimic_motion_state(
            0, self.motion_times[env_id].unsqueeze(0)
        )
        ref_gt = ref_state.rb_pos
        ref_gr = ref_state.rb_rot
        ref_lr = ref_state.local_rot
        ref_gv = ref_state.rb_vel
        ref_gav = ref_state.rb_ang_vel
        ref_dv = ref_state.dof_vel

        ref_lr = ref_lr[:, self.dof_body_ids]
        ref_kb = self.process_kb(ref_gt, ref_gr)

        gt = torch.from_numpy(self.mjc_datas[env_id].xpos[1:].copy()).to(dtype=torch.float32,
                                                                      device=self.device).unsqueeze(0)
        gr = torch.from_numpy(self.mjc_datas[env_id].xquat[1:].copy()).to(dtype=torch.float32,
                                                                       device=self.device).unsqueeze(0)
        gv = torch.from_numpy(self.mjc_datas[env_id].cvel[1:, 3:].copy()).to(dtype=torch.float32,
                                                                          device=self.device).unsqueeze(0)
        gav = torch.from_numpy(self.mjc_datas[env_id].cvel[1:, :3].copy()).to(dtype=torch.float32,
                                                                              device=self.device).unsqueeze(0)

        gr = gr[:, :, rot_conv_mjc_to_isaac]

        """
        # first remove height based on current position
        gt[:, :, -1:] -= self.get_ground_heights(gt[:, 0, :2]).view(self.num_envs, 1, 1)
        # then remove offset to get back to the ground-truth data position
        gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(
            self.num_envs, 1, 2
        )
        """

        kb = self.process_kb(gt, gr)

        rt = gt[:, 0]
        ref_rt = ref_gt[:, 0]

        if self.config.mimic_reward_config.rt_ignore_height:
            rt = rt[..., :2]
            ref_rt = ref_rt[..., :2]

        rr = gr[:, 0]
        ref_rr = ref_gr[:, 0]

        inv_heading = torch_utils.calc_heading_quat_inv(rr, True)
        ref_inv_heading = torch_utils.calc_heading_quat_inv(ref_rr, True)

        rv = gv[:, 0]
        ref_rv = ref_gv[:, 0]

        rav = gav[:, 0]
        ref_rav = ref_gav[:, 0]

        dp, dv = self.get_dof_state(env_id)
        lr = mimic_utils.dof_to_local(dp, self.dof_offsets, True)

        if self.config.mimic_reward_config.add_rr_to_lr:
            rr = gr[:, 0]
            ref_rr = ref_gr[:, 0]

            lr = torch.cat([rr.unsqueeze(1), lr], dim=1)
            ref_lr = torch.cat([ref_rr.unsqueeze(1), ref_lr], dim=1)

        rew_dict = mimic_utils.exp_tracking_reward(
            gt=gt,
            rt=rt,
            kb=kb,
            gr=gr,
            lr=lr,
            rv=rv,
            rav=rav,
            gv=gv,
            gav=gav,
            dv=dv,
            ref_gt=ref_gt,
            ref_rt=ref_rt,
            ref_kb=ref_kb,
            ref_gr=ref_gr,
            ref_lr=ref_lr,
            ref_rv=ref_rv,
            ref_rav=ref_rav,
            ref_gv=ref_gv,
            ref_gav=ref_gav,
            ref_dv=ref_dv,
            joint_reward_weights=1, # Return uniform weights if unequal weighting is disabled or not using SMPLX model
            config=self.config.mimic_reward_config,
            w_last=True,
        )

        """
        current_contact_forces = self.get_bodies_contact_buf()
        forces_delta = torch.clip(
            self.prev_contact_forces - current_contact_forces, min=0
        )[
                       :, self.non_termination_contact_body_ids, 2
                       ]  # get the Z axis
        kbf_rew = (
            forces_delta.sum(-1)
            .mul(self.config.mimic_reward_config.component_coefficients.kbf_rew_c)
            .exp()
        )

        rew_dict["kbf_rew"] = kbf_rew

        dof_forces = self.get_dof_forces()
        power = torch.abs(torch.multiply(dof_forces, dv)).sum(dim=-1)
        pow_rew = -power

        has_reset_grace = (
                self.reset_track_steps.steps <= self.config.mimic_reset_track.grace_period
        )
        pow_rew[has_reset_grace] = 0

        rew_dict["pow_rew"] = pow_rew
        """

        self.last_scaled_rewards[env_id] = {
            k: v * getattr(self.config.mimic_reward_config.component_weights, f"{k}_w")
            for k, v in rew_dict.items()
        }

        tracking_rew = sum(self.last_scaled_rewards[env_id].values())


        translation_mask_coeff = self.config.robot.num_bodies
        rotation_mask_coeff = self.config.robot.num_bodies

        gt_err = (ref_gt - gt).pow(2).sum(-1).sqrt().sum(-1).div(translation_mask_coeff)
        max_joint_err = (ref_gt - gt).pow(2).sum(-1).sqrt().max(-1)[0]

        gr_diff = humanoid_utils.quat_diff_norm(gr, ref_gr, True)
        gr_err = gr_diff.sum(-1).div(rotation_mask_coeff)
        gr_err_degrees = gr_err * 180 / torch.pi
        max_gr_err = gr_diff.max(-1)[0]
        max_gr_err_degrees = max_gr_err * 180 / torch.pi

        lr_diff = humanoid_utils.quat_diff_norm(lr, ref_lr, True)
        lr_err = lr_diff.sum(-1).div(rotation_mask_coeff)
        lr_err_degrees = lr_err * 180 / torch.pi
        max_lr_err = lr_diff.max(-1)[0]
        max_lr_err_degrees = max_lr_err * 180 / torch.pi

        self.last_other_rewards[env_id] = {
            "tracking_rew": tracking_rew,
            "gt_err": gt_err,
            "gr_err": gr_err,
            "gr_err_degrees": gr_err_degrees,
            "lr_err_degrees": lr_err_degrees,
            "max_joint_err": max_joint_err,
            "max_lr_err_degrees": max_lr_err_degrees,
            "max_gr_err_degrees": max_gr_err_degrees,
        }

        return tracking_rew + self.config.mimic_reward_config.positive_constant

    def process_kb(self, gt: Tensor, gr: Tensor):
        kb = gt[:, self.key_body_ids]

        if self.config.mimic_reward_config.relative_kb_pos:
            rt = gt[:, 0]
            rr = gr[:, 0]
            kb = kb - rt.unsqueeze(1)

            heading_rot = torch_utils.calc_heading_quat_inv(rr, True)
            rr_expand = heading_rot.unsqueeze(1).expand(rr.shape[0], kb.shape[1], 4)
            kb = rotations.quat_rotate(
                rr_expand.reshape(-1, 4), kb.view(-1, 3), True
            ).view(kb.shape)

        return kb

    def get_dof_state(self, env_id):
        joint_offset = 3 + 4

        return (torch.from_numpy(self.mjc_datas[env_id].qpos[joint_offset:joint_offset + self.num_act]).to(self.device).unsqueeze(0),
                torch.from_numpy(self.mjc_datas[env_id].qvel[joint_offset - 1:joint_offset - 1 + self.num_act]).to(self.device).unsqueeze(0))

    def _compute_obs(self, env_id):

        body_pos = torch.from_numpy(self.mjc_datas[env_id].xpos[1:].copy()).to(dtype=torch.float32,
                                                                      device=self.device).unsqueeze(0)
        body_rot = torch.from_numpy(self.mjc_datas[env_id].xquat[1:].copy()).to(dtype=torch.float32,
                                                                       device=self.device).unsqueeze(0)
        body_vel = torch.from_numpy(self.mjc_datas[env_id].cvel[1:, 3:].copy()).to(dtype=torch.float32,
                                                                          device=self.device).unsqueeze(0)
        body_ang_vel = torch.from_numpy(self.mjc_datas[env_id].cvel[1:, :3].copy()).to(dtype=torch.float32,
                                                                              device=self.device).unsqueeze(0)

        body_rot = body_rot[:,:, rot_conv_mjc_to_isaac]

        cat_obs = humanoid_utils.compute_humanoid_observations_max(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            torch.zeros(1, device=self.device).unsqueeze(0),
            True,  # self.config.humanoid_obs.local_root_obs
            True,  # self.config.humanoid_obs.root_height_obs
            True,  # self.w_last
        )

        cur_gt = body_pos
        cur_gr = body_rot

        num_future_steps = 5

        time_offsets = (
                torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
                * (1.0 / 30.0)
        )

        raw_future_times = self.motion_times[env_id].unsqueeze(-1) + time_offsets.unsqueeze(0)
        flat_ids = torch.tensor(0, device=self.device)

        lengths = self.motion_lib.get_motion_length(flat_ids)
        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.motion_lib.get_mimic_motion_state(flat_ids, flat_times)
        flat_target_pos = ref_state.rb_pos
        #flat_target_rot = ref_state.rb_rot[...,rot_conv_mjc_to_isaac]

        mimic_target = mimic_utils.build_max_coords_target_poses_future_rel(
            cur_gt=cur_gt,
            cur_gr=cur_gr,
            flat_target_pos=flat_target_pos,
            flat_target_rot=ref_state.rb_rot,
            num_envs=1,
            num_future_steps=num_future_steps,
            mimic_conditionable_bodies_ids=torch.arange(17, dtype=torch.long, device=self.device
                                                        ),
            w_last=True,
        )

        return cat_obs, mimic_target

    def setup_character_props(self):
        self.dof_body_ids = self.config.robot.dfs_dof_body_ids
        self.dof_offsets = []
        previous_dof_name = "null"
        for dof_offset, dof_name in enumerate(self.config.robot.dfs_dof_names):
            if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
                previous_dof_name = dof_name[:-2]
                self.dof_offsets.append(dof_offset)
        self.dof_offsets.append(len(self.config.robot.dfs_dof_names))
        self.dof_obs_size = self.config.robot.dof_obs_size
        self.num_act = self.config.robot.number_of_actions