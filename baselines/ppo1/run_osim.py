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

def train(num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank != 0: logger.set_level(logger.DISABLED)
    workerseed = seed + 1000000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = FixedRunEnv(visualize=True)

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    #env = bench.Monitor(env, logger.get_dir() and 
    #    osp.join(logger.get_dir(), "monitor.json"))
    env.seed(workerseed)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(num_timesteps=1e6, seed=args.seed)


if __name__ == '__main__':
    main()
