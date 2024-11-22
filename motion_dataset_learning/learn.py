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

config = yaml.safe_load(open('motion_dataset_learning/config.yml'))

logging.basicConfig(level=logging.INFO, filename="py_log.log",filemode="a")

learned_motions = [name for name in os.listdir(config['learned_models_root'])]

nonlearned_motions = [name.split('.')[0] for name in os.listdir(config['motions_root']) if
                          name.split('.')[0] not in learned_motions]

while len(nonlearned_motions) > 0:

    motion_name = nonlearned_motions.pop(0)

    logging.info(f'learn {motion_name}\n')

    # preprate to run
    motion_file = config['motions_root'] + '/' + motion_name + '.npy'
    train_script = config['train_script']
    python_path = config['python_path']
    cmd = f'{python_path} {train_script} +exp=h1_full_body_tracker +backbone=isaacgym +robot=sword_and_shield motion_file={motion_file} num_envs=512'

    # run
    train_process = subprocess.Popen(cmd, shell=True)
    event_acc = event_accumulator.EventAccumulator(config['log_dir'])

    # watch process
    # early_stopping = EarlyStopping(patience=config['patience'], min_delta=config['min_delta'])
    start_time = time.time()
    while True:
        time.sleep(30)
        event_acc.Reload()
        rewards_events = event_acc.Scalars('rewards/total_rewards')

        rewards = [reward.value for reward in rewards_events]

        if len(rewards) < config['patience'] + 1:
            continue

        best = max(rewards[:-config['patience']])

        mean_on_patience = np.mean(rewards[:-config['patience']])
        std_on_patience = np.std(rewards[-config['patience']:])

        #early_stopping.reset()
        #for reward in rewards_events[-config['patience'] - 1:]:
        #    early_stopping(reward.value)

        if time.time() - start_time >= 60 * config['one_model_max_train_time_minutes']:
            logging.info(f'Early stop by train time. best: {best}, std_on_patience: {std_on_patience}, mean_on_patience: {mean_on_patience}')
            kill_all(train_process.pid)
            break

        if std_on_patience < config['min_delta'] and  mean_on_patience > best:
            logging.info(f'Early stop by std. best: {best}, std_on_patience: {std_on_patience}, mean_on_patience: {mean_on_patience}')
            kill_all(train_process.pid)
            break

    # move rename
    shutil.move('/home/damibran/dev/repos/ProtoMotions/results/h1_full_body_tracker/lightning_logs/version_0', config['learned_models_root'])
    os.rename(config['learned_models_root']+'/'+'version_0', config['learned_models_root'] +'/'+ motion_name)

    # iterate
    learned_motions = [name for name in os.listdir(config['learned_models_root'])]

    nonlearned_motions = [name.split('.')[0] for name in os.listdir(config['motions_root']) if
                          name.split('.')[0] not in learned_motions]