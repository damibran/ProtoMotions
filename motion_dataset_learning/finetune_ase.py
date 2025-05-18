import shutil

import yaml
import os
import subprocess
import psutil
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tensorboard.backend.event_processing import event_accumulator
from distutils.dir_util import copy_tree
import numpy as np
import logging
import time

python_path = "/home/damibran/anaconda3/envs/ProtoMotionsEnv/bin/python"
train_script = "/home/damibran/dev/repos/ProtoMotionsFork/phys_anim/train_agent.py"

checkpoints = [
    "/home/damibran/dev/repos/ProtoMotionsFork/results/iql/lightning_logs/version_0/last.ckpt_90",
    "/home/damibran/dev/repos/ProtoMotionsFork/results/iql_calm/lightning_logs/version_0/last.ckpt",
    "/home/damibran/dev/repos/ProtoMotionsFork/results/iql_fenc/lightning_logs/version_6_actor_enc_join_loss/last.ckpt_90",
    "/home/damibran/dev/repos/ProtoMotionsFork/results/iql_fenc/lightning_logs/version_8/last.ckpt_90"
]

for checkpoint in checkpoints:
    cmd = f'{python_path} {train_script} +exp=ase_actions +robot=sword_and_shield +backbone=isaacgym +motion_file=output/recordings/IQL_seconddataset/sword_shield_state_action_reduce.yaml num_envs=2048 +checkpoint={checkpoint}'
    train_process = subprocess.Popen(cmd, shell=True)
    train_process.wait()