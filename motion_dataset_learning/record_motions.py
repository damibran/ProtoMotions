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

def kill_all(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

config = yaml.safe_load(open('motion_dataset_learning/config_record.yml'))

to_record = list(config['to_record_list'])

while len(to_record)>0:
    motion_name = to_record.pop(0)

    print(f'/////////////////////////////////////////////////////   {motion_name}   ///////////////////////////////////////////////////')


    check_point_path = config['learned_models_root'] + '/' + motion_name + '/last.ckpt'

    # preprate to run
    motion_file = config['motions_root'] + '/' + motion_name + '.npy'
    eval_script = config['eval_script']
    python_path = config['python_path']
    cmd = f'{python_path} {eval_script} +exp=record_motion +robot=sword_and_shield +backbone=isaacgym +motion_file={motion_file} +checkpoint={check_point_path} +headless=False'

    # run
    train_process = subprocess.Popen(cmd, shell=True)

    train_process.wait()

    config['to_record_list'] = to_record
    yaml.safe_dump(config, open('motion_dataset_learning/config_record.yml','w'))