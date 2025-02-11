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
from utils import motion_lib
from utils.motion_lib import MotionLib


def kill_all(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


config = yaml.safe_load(open('motion_dataset_learning/config_record.yml'))

to_record = list(config['to_record_list'])

while len(to_record) > 0:
    motion_name = to_record.pop(0)
    i = 1
    while i < config["each_motion_count"]:
        print(
            f'/////////////////////////////////////////////////////   {motion_name}   ///////////////////////////////////////////////////')

        check_point_path = config['learned_models_root'] + '/' + motion_name + '/last.ckpt'

        # preprate to run
        motion_file = config['motions_root'] + '/' + motion_name + '.npy'
        eval_script = config['eval_script']
        python_path = config['python_path']
        cmd = f'{python_path} {eval_script} +exp=record_motion +robot=sword_and_shield +backbone=isaacgym +motion_file={motion_file} +checkpoint={check_point_path} +headless=False'

        # run
        train_process = subprocess.Popen(cmd, shell=True)

        train_process.wait()

        s_lib = MotionLib(motion_file,
                          [1, 2,3,4,5,7,8,11,12,13,14,15,16],
                          [0,3,6,9,10,13,16,17,20,21,24,27,28,31],
                            []
                          )

        sa_lib = MotionLib(config["recordings_root"] + '/' + f"{motion_name}.npy",
                           [1, 2,3,4,5,7,8,11,12,13,14,15,16],
                           [0,3,6,9,10,13,16,17,20,21,24,27,28,31],
                           []
                           )

        if abs(s_lib.state.motion_lengths[0] - sa_lib.state.motion_lengths[0]) < 0.1:
            os.rename(config["recordings_root"] + '/' + f"{motion_name}.npy",
                      config["recordings_root"] + '/' + f"{motion_name}_{i}.npy")
            os.rename(config["recordings_root"] + '/' + f"{motion_name}_actions.npy",
                      config["recordings_root"] + '/' + f"{motion_name}_actions_{i}.npy")
        else:
            i -= 1

        i += 1

    config['to_record_list'] = to_record
    yaml.safe_dump(config, open('motion_dataset_learning/config_record.yml', 'w'))
