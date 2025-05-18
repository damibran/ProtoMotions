# ProtoMotions IQL Fork

This repository contains our experiments on learning ASE and CALM low level controllers with offline reinforcement learning ([IQL](https://github.com/ikostrikov/implicit_q_learning) algorithm). And our method FENC (future encoder).

Here we have:
  - `full_body_tracker_record` experiment which records actor observations, actions, rewards, dones in `hdf5` both in train time and eval and can be used along with `motion_dataset_learning/learn_train_dataset` to collect whole motion dataset
  - `iql_exp` expiriment to train IQL ASE (ASE, was sligtly modified: to prevent constant likelihood, we'v added actions to encoder\discriminator input)
  - `iql_calm_exp` experiment to train IQL CALM
  - `iql_fenc_exp` experiment to train our FENC model
  - additional `iql_mimic_exp` experiment to train full_body_tracker in offline, which we've used to verify our training routine
  - `motion_dataset_learning/motion_classifier` along with `ase_actions_inception` can be used to evaluate inception score of trained controllers
  - `motion_dataset_learning/fisher_distance.py` can be used to calculate encoder fisher distance
  - and other useful scripts...