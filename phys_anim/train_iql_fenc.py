import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

has_robot_arg = False
backbone = None
for arg in sys.argv:
    # This hack ensures that isaacgym is imported before any torch modules.
    # The reason it is here (and not in the main func) is due to pytorch lightning multi-gpu behavior.
    if "robot" in arg:
        has_robot_arg = True
    if "backbone" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +backbone")
        if "isaacgym" in arg.split("=")[-1]:
            import isaacgym  # noqa: F401

            backbone = "isaacgym"
        elif "isaacsim" in arg.split("=")[-1]:
            from isaacsim import SimulationApp
            from phys_anim.envs.base_interface.isaacsim_utils.experiences import (
                get_experience,
            )

            backbone = "isaacsim"

from phys_anim.agents.iql_fenc import IQL_Fenc

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from utils.config_utils import *  # noqa: E402, F403
from utils.common import seeding

from isaac_utils import torch_utils

from phys_anim.agents.callbacks.slurm_autoresume import (
    SlurmAutoResume,
)  # noqa: E402


def build_body_ids_tensor(self, body_names):
    body_ids = []

    for body_name in body_names:
        body_id = self.body_names.index(body_name)
        assert (
                body_id != -1
        ), f"Body part {body_name} not found in {self.body_names}"
        body_ids.append(body_id)

    body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)
    return body_ids

@hydra.main(config_path="config", config_name="iql_base")
def main(config: OmegaConf):
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    autoresume = SlurmAutoResume()
    id = autoresume.details.get("id")
    if (
        id is not None
        and "wandb" in config
        and OmegaConf.select(config, "wandb.wandb_id", default=None) is None
    ):
        config = OmegaConf.merge(config, OmegaConf.create({"wandb": {"wandb_id": id}}))

    torch.set_float32_matmul_precision("medium")

    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if config.seed is not None:
        rank = fabric.global_rank
        if rank is None:
            rank = 0
        fabric.seed_everything(config.seed + rank)
        seeding(config.seed + rank, torch_deterministic=config.torch_deterministic)

    # Env, Humanoid substitute
    algo: IQL_Fenc = IQL_Fenc(fabric=fabric, config=config)
    algo.setup()
    if config.auto_load_latest:
        latest_checkpoint = Path(fabric.loggers[0].root_dir) / "last.ckpt"
        if latest_checkpoint.exists():
            config.checkpoint = latest_checkpoint
    #algo.load(config.checkpoint)

    save_dir = Path(fabric.loggers[0].log_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving config file to {save_dir}")
    with open(save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)

    algo.fit()
    #algo.fit_encoder()
    #algo.fit_discriminator()


if __name__ == "__main__":
    main()