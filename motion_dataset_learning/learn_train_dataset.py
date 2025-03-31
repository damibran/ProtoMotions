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
import time

def kill_all(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

config = yaml.safe_load(open('motion_dataset_learning/config_train_dataset.yml'))

nonlearned_motions = list(config['motions_to_train'])

while len(nonlearned_motions) > 0:

    motion_name = nonlearned_motions.pop(0)

    # preprate to run
    motion_file = config['motions_root'] + '/' + motion_name + '.npy'
    train_script = config['train_script']
    python_path = config['python_path']
    cmd = f'{python_path} {train_script} +exp=full_body_tracker_record +robot=sword_and_shield +backbone=isaacgym motion_file={motion_file} num_envs=512'

    # run
    train_process = subprocess.Popen(cmd, shell=True)
    event_acc = event_accumulator.EventAccumulator(config['version_0_results_dir'])

    train_process.wait()

    # move rename
    os.rename(config['results_dir']+'/'+'version_0', config['results_dir'] +'/'+ motion_name)

    # iterate

    config['motions_to_train'] = nonlearned_motions
    yaml.safe_dump(config, open('motion_dataset_learning/config_train_dataset.yml', 'w'))