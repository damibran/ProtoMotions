import torch
from torch import Tensor
from phys_anim.utils.motion_lib import MotionLib
from phys_anim.utils.running_mean_std import RunningMeanStd
from phys_anim.envs.humanoid.humanoid_utils import build_disc_observations
from phys_anim.agents.models.mlp import build_mlp
from omegaconf import OmegaConf

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def align_loss(x, y, alpha=2):
    return torch.linalg.norm(x - y, ord=2, dim=1).pow(alpha).mean()

class CalmEncoderTrainer:
    def __init__(self):
        self._calm_latent_shape = 64

        self.motion_file = 'phys_anim/data/motions/sword_shield/dataset_reallusion_sword_shield.yaml'

        robot_dfs_dof_body_names = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'sword', 'left_upper_arm', 'left_lower_arm', 'shield', 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']

        self.dof_body_ids = [ 1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16 ] #self.all_config.robot.dfs_dof_body_ids
        self.dof_offsets = []
        previous_dof_name = "null"
        for dof_offset, dof_name in enumerate(robot_dfs_dof_body_names): #self.all_config.robot.dfs_dof_names
            if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
                previous_dof_name = dof_name[:-2]
                self.dof_offsets.append(dof_offset)
        self.dof_offsets.append(len(robot_dfs_dof_body_names)) #self.all_config.robot.dfs_dof_names
        self.dof_obs_size = 78 #self.all_config.robot.dof_obs_size
        self.num_act = 31 #self.all_config.robot.number_of_actions

        key_bodies = [ "sword", "shield"]

        self.key_body_ids = torch.tensor(
            [
                robot_dfs_dof_body_names.index(key_body_name) #self.all_config.robot.dfs_body_names.index(key_body_name)
                for key_body_name in key_bodies #self.all_config.robot.key_bodies
            ],
            dtype=torch.long,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._motion_lib = MotionLib(
                motion_file=self.motion_file,
                dof_body_ids=self.dof_body_ids,
                dof_offsets=self.dof_offsets,
                key_body_ids=self.key_body_ids,
                device=self.device,
            )

        self._num_amp_obs_steps = 10
        self._num_amp_obs_enc_steps = 60

        self.dt = 1.0/30.0

        self._amp_minibatch_size = 4096

        self._normalize_amp_input = True

        num_key_bodies = len(key_bodies)

        self._num_amp_obs_per_step = 13 + self.dof_obs_size + 31 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(shape = (self._num_amp_obs_per_step // self._num_amp_obs_steps,)).to(self.device)

        mlp_conf = OmegaConf.create()

        mlp_conf['units'] = [ 1024, 1024, 512 ]
        mlp_conf['activation'] = 'relu'
        mlp_conf['initializer'] = 'default'
        mlp_conf['use_layer_norm'] = False

        self._enc = build_mlp(mlp_conf, self._num_amp_obs_per_step * self._num_amp_obs_enc_steps, self._calm_latent_shape)

        pass

        for i in range(len(self._motion_lib.state.motion_lengths)):
            if self._motion_lib.state.motion_lengths[i] < 2.0:
                print(self._motion_lib.state.motion_files[i])

        pass

    def _enc_reg_loss(self):
        enc_amp_obs_demo, _ = self._fetch_amp_obs_demo(self._amp_minibatch_size)
        proc_enc_amp_obs_demo = self._preproc_amp_obs(enc_amp_obs_demo)

        amp_obs_encoding = self._enc(proc_enc_amp_obs_demo)

        # Loss for uniform distribution over the sphere
        uniform_l = uniform_loss(amp_obs_encoding)

        _, _, similar_enc_amp_obs_demo0, _, similar_enc_amp_obs_demo1 = self.fetch_amp_obs_demo_pair(self._amp_minibatch_size)

        proc_similar_enc_amp_obs_demo0 = self._preproc_amp_obs(similar_enc_amp_obs_demo0)
        proc_similar_enc_amp_obs_demo1 = self._preproc_amp_obs(similar_enc_amp_obs_demo1)

        similar_amp_obs_encoding0 = self._enc(proc_similar_enc_amp_obs_demo0)
        similar_amp_obs_encoding1 = self._enc(proc_similar_enc_amp_obs_demo1)

        # Loss for alignment - overlapping motions should have 'close' embeddings
        align_l = align_loss(similar_amp_obs_encoding0, similar_amp_obs_encoding1)

        loss = align_l + 0.5 * uniform_l

        return {'enc_reg_loss': loss}

    def _fetch_amp_obs_demo(self, num_samples):
        _, _, enc_amp_obs_demo_flat, _, amp_obs_demo_flat = self.fetch_amp_obs_demo_enc_pair(num_samples)
        return enc_amp_obs_demo_flat, amp_obs_demo_flat

    def fetch_amp_obs_demo_enc_pair(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        enc_motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=enc_window_size)
        # make sure not to add more than motion clip length, negative amp_obs will show zero index amp_obs instead
        enc_motion_times += torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)

        # sub-window-size is for the amp_obs contained within the enc-amp-obs. make sure we sample only within the valid portion of the motion
        sub_window_size = torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size) - self.dt * self._num_amp_obs_steps
        motion_times = enc_motion_times - torch.rand(enc_motion_times.shape, device=self.device) * sub_window_size

        enc_amp_obs_demo = self.build_disc_obs_demo(motion_ids, enc_motion_times).view(-1, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step)
        amp_obs_demo = self.build_disc_obs_demo(motion_ids, motion_times).view(-1, self._num_amp_obs_steps, self._num_amp_obs_per_step)

        enc_amp_obs_demo_flat = enc_amp_obs_demo.to(self.device).view(-1, self.get_num_enc_amp_obs())
        amp_obs_demo_flat = amp_obs_demo.to(self.device).view(-1, self.get_num_amp_obs())

        return motion_ids, enc_motion_times, enc_amp_obs_demo_flat, motion_times, amp_obs_demo_flat

    def fetch_amp_obs_demo_pair(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)
        cat_motion_ids = torch.cat((motion_ids, motion_ids), dim=0)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=enc_window_size)
        motion_times0 += torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)

        motion_times1 = motion_times0 + torch.rand(motion_times0.shape, device=self._motion_lib._device) * 0.5
        motion_times1 = torch.min(motion_times1, self._motion_lib._motion_lengths[motion_ids])

        motion_times = torch.cat((motion_times0, motion_times1), dim=0)

        amp_obs_demo = self.build_disc_obs_demo(cat_motion_ids, motion_times).view(-1, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step)
        amp_obs_demo0, amp_obs_demo1 = torch.split(amp_obs_demo, num_samples)

        amp_obs_demo0_flat = amp_obs_demo0.to(self.device).view(-1, self.get_num_enc_amp_obs())

        amp_obs_demo1_flat = amp_obs_demo1.to(self.device).view(-1, self.get_num_enc_amp_obs())

        return motion_ids, motion_times0, amp_obs_demo0_flat, motion_times1, amp_obs_demo1_flat

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            shape = amp_obs.shape
            amp_obs = amp_obs.view(-1, self._num_amp_obs_per_step // self._num_amp_obs_steps)
            amp_obs = self._amp_input_mean_std(amp_obs)
            amp_obs = amp_obs.view(shape)
        return amp_obs

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def get_num_enc_amp_obs(self):
        return self._num_amp_obs_enc_steps * self._num_amp_obs_per_step

    def build_disc_obs_demo(self, motion_ids: Tensor, motion_times0: Tensor):
        dt = self.dt

        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps]
        )
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(
            0, self._num_amp_obs_steps, device=self.device
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        # motion ids above are "sub_motions" so first we map to motion file itself and then extract the length.
        lengths = self._motion_lib.state.motion_lengths[
            self._motion_lib.state.sub_motion_to_motion[motion_ids]
        ]

        assert torch.all(motion_times >= 0)
        assert torch.all(motion_times <= lengths)

        ref_state = self._motion_lib.get_motion_state(motion_ids, motion_times)

        disc_obs_demo = build_disc_observations(
            ref_state.root_pos,
            ref_state.root_rot,
            ref_state.root_vel,
            ref_state.root_ang_vel,
            ref_state.dof_pos,
            ref_state.dof_vel,
            ref_state.key_body_pos,
            torch.zeros(len(motion_ids), 1, device=self.device),
            True, #self.config.humanoid_obs.local_root_obs
            True,
            self.dof_obs_size,
            self.get_dof_offsets(),
            False,
            True,
        )
        return disc_obs_demo

    def get_dof_offsets(self):
        return self.dof_offsets

if __name__ == '__main__':
    trainer = CalmEncoderTrainer()

    trainer._enc_reg_loss()