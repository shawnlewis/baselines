#!/usr/bin/env python

import opensim

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
from osim.env import RunEnv
import numpy as np
import math
import time

class FixedRunEnv(RunEnv):
    def __init__(self, visualize=False, difficulty=2,
            fall_smoothing_exp=2, fall_smoothing_max_penalty=0, velocity_penalty_mult=0,
            alive_bonus=0, time_penalty=0):
        self._saved_difficulty = difficulty
        self._fall_smoothing_exp = fall_smoothing_exp
        self._fall_smoothing_max_penalty = fall_smoothing_max_penalty
        self._velocity_penalty_mult = velocity_penalty_mult
        self._alive_bonus = alive_bonus
        self._time_penalty = time_penalty
        super(FixedRunEnv, self).__init__(visualize=visualize)
        self.spec.timestep_limit = 500
        self.horizon = 500

    def seed(self, seed):
        self._saved_seed = seed
    
    def reset(self):
        self._saved_seed += 1
        return np.array(super(FixedRunEnv, self).reset(
            difficulty=self._saved_difficulty, seed=self._saved_seed))

    def get_observation(self):
        return np.array(super(FixedRunEnv, self).get_observation())

    def step(self, action):
        start_time = time.time()
        next_obs, reward, done, info = super(FixedRunEnv, self).step(action)
        step_time = time.time() - start_time

        # Penalize for pelvis height getting close to 0.65.
        pelvis_y = self.current_state[self.STATE_PELVIS_Y]
        if pelvis_y < 1:
            # Quadratically go from MAX_PENALTY at pelvis_y=0.65 (simulation ends) to
            # 0 at pelvis_y=1.0 which is the max
            # Setting EXPONENT to 1 makes this linear
            reward -= self._fall_smoothing_max_penalty * (-(pelvis_y - 1)/.35) ** self._fall_smoothing_exp

        # Penalyze fast moving body parts
        velocities = np.array(self.current_state[-19:-5]) - np.array(self.last_state[-19:-5])
        vel_squared_sum = np.sum(np.square(velocities))
        reward -= self._velocity_penalty_mult * vel_squared_sum

        reward += self._alive_bonus

        reward -= self._time_penalty * step_time

        return next_obs, reward, done, info

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

    env = FixedRunEnv(
        visualize=config['visualize'],
        difficulty=config['difficulty'],
        fall_smoothing_exp=config['fall_smoothing_exp'],
        fall_smoothing_max_penalty=config['fall_smoothing_max_penalty'],
        velocity_penalty_mult=config['velocity_penalty_mult'],
        alive_bonus=config['alive_bonus'],
        time_penalty=config['time_penalty'])

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_sizes=config['hidden_layers'],
            num_hid_layers=len(config['hidden_layers']),
            gaussian_fixed_var=True,
            init_pol_weight_stddev=config['init_pol_weight_stddev'],
            init_logstd=config['init_logstd'])
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
            load_model=config['load_model'],
            action_bias=config['action_bias'],
            action_repeat=config['action_repeat'],
            action_repeat_rand=config['action_repeat_rand'],
            target_kl=config['target_kl'],
            warmup_frames=config['warmup_frames'],
            vf_loss_mult=config['vf_loss_mult']
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
