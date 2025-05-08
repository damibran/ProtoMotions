import torch

from torch import Tensor
import math

from lightning.fabric import Fabric

from hydra.utils import instantiate
from isaac_utils import torch_utils

from phys_anim.agents.utils.data_utils import swap_and_flatten01
from phys_anim.agents.ppo import PPO, get_params
from phys_anim.utils.replay_buffer import ReplayBuffer
from phys_anim.utils.dataset import GeneralizedDataset
from phys_anim.agents.models.discriminator import JointDiscMLP
from phys_anim.envs.amp.common import DiscHumanoid
from phys_anim.agents.models.actor import PPO_Actor
from pathlib import Path

class ASEPPO(PPO):
    def __init__(self, fabric: Fabric, env, config):
        super().__init__(fabric, env, config)
        delattr(self.experience_buffer,"mus")
        delattr(self.experience_buffer, "sigmas")
        delattr(self.experience_buffer, "actions")
        latent_dim = sum(self.config.infomax_parameters.latent_dim)
        self.experience_buffer.register_key("mus", shape=(latent_dim,))
        self.experience_buffer.register_key("sigmas", shape=(latent_dim,))
        self.experience_buffer.register_key("actions", shape=(latent_dim,))

    def setup(self):
        super().setup()

        actor: PPO_Actor = instantiate(
            self.config.actor, num_in=self.num_obs, num_act=sum(self.config.infomax_parameters.latent_dim)
        )
        actor_optimizer = instantiate(
            self.config.actor_optimizer,
            params=list(actor.parameters()),
            _convert_="all",
        )

        self.actor, self.actor_optimizer = self.fabric.setup(actor, actor_optimizer)
        self.actor.mark_forward_method("eval_forward")
        self.actor.mark_forward_method("training_forward")

        llc: PPO_Actor = instantiate(
            self.config.llc, num_in=self.num_obs, num_act=self.num_act
        )
        llc_checkpoint = Path(self.config.llc.config.checkpoint).resolve()
        print(f"Loading LLC model from checkpoint: {llc_checkpoint}")
        state_dict = torch.load(llc_checkpoint) #, map_location=self.device
        llc.load_state_dict(state_dict['actor'])
        self.llc = llc.to(self.device)
        self.llc.eval()
        self.llc.requires_grad = False

    def env_step(self, actor_state):
        actor_state["latents"] = actor_state["actions"]
        actions = self.llc.eval_forward(actor_state)["actions"]
        obs, rewards, dones, extras = self.env.step(actions)
        rewards = rewards * self.task_reward_w
        actor_state.update(
            {"obs": obs, "rewards": rewards, "dones": dones, "extras": extras}
        )

        actor_state = self.get_extra_obs_from_env(actor_state)

        return actor_state