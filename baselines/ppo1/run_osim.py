#!/usr/bin/env python

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
from osim.env import RunEnv
import numpy as np

class FixedRunEnv(RunEnv):
    def __init__(self, visualize=False, difficulty=2):
        self._saved_difficulty = difficulty
        super(FixedRunEnv, self).__init__(visualize=visualize)

    def seed(self, seed):
        self._saved_seed = seed
    
    def reset(self):
        self._saved_seed += 1
        return np.array(super(FixedRunEnv, self).reset(
            difficulty=self._saved_difficulty, seed=self._saved_seed))

    def get_observation(self):
        return np.array(super(FixedRunEnv, self).get_observation())

def train():
    from baselines.ppo1 import mlp_policy, pposgd_simple
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        import wandb
        run = wandb.init()
        config = run.config._items
        print(config)
    else:
        config = None
    config = MPI.COMM_WORLD.bcast(config)

    sess = U.single_threaded_session()
    sess.__enter__()

    if rank != 0: logger.set_level(logger.DISABLED)

    workerseed = config['seed'] + 1000000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = FixedRunEnv(visualize=config['visualize'])

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_sizes=config['hidden_layers'],
            num_hid_layers=len(config['hidden_layers']))
    #env = bench.Monitor(env, logger.get_dir() and 
    #    osp.join(logger.get_dir(), "monitor.json"))
    env.seed(workerseed)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=config['max_timesteps'],
            timesteps_per_batch=config['horizon'],
            clip_param=config['clip_param'], entcoeff=config['entcoeff'],
            optim_epochs=config['optim_epochs'], optim_stepsize=config['optim_stepsize'],
            optim_batchsize=config['optim_batchsize'],
            gamma=config['discount'], lam=config['lam'], schedule=config['lr_schedule'],
            load_model=config['load_model']
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train()


if __name__ == '__main__':
    main()
