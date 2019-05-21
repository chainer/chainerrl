from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import os
import random

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

from chainer import links as L
import gym
from gym import spaces
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl.agents import nsq
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers import rmsprop_async

from chainerrl.wrappers import atari_wrappers


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--final-exploration-frames',
                        type=int, default=4 * 10 ** 6)
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--eval-interval', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = process_seeds[process_idx]
        env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, 0.05)
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    sample_env = make_env(0, test=False)
    action_space = sample_env.action_space
    assert isinstance(action_space, spaces.Discrete)

    # Define a model and its optimizer
    q_func = links.Sequence(
        links.NIPSDQNHead(),
        L.Linear(256, action_space.n),
        DiscreteActionValue)
    opt = rmsprop_async.RMSpropAsync(lr=args.lr, eps=1e-1, alpha=0.99)
    opt.setup(q_func)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    # Make process-specific agents to diversify exploration
    def make_agent(process_idx):
        # Random epsilon assignment described in the original paper
        rand = random.random()
        if rand < 0.4:
            epsilon_target = 0.1
        elif rand < 0.7:
            epsilon_target = 0.01
        else:
            epsilon_target = 0.5
        explorer = explorers.LinearDecayEpsilonGreedy(
            1, epsilon_target, args.final_exploration_frames,
            action_space.sample)
        # Suppress the explorer logger
        explorer.logger.setLevel(logging.INFO)
        return nsq.NSQ(q_func, opt, t_max=5, gamma=0.99,
                       i_target=40000,
                       explorer=explorer, phi=phi)

    if args.demo:
        env = make_env(0, True)
        agent = make_agent(0)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            agent.optimizer.lr = value

        lr_decay_hook = experiments.LinearInterpolationHook(
            args.steps, args.lr, 0, lr_setter)

        experiments.train_agent_async(
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            make_agent=make_agent,
            profile=args.profile,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            global_step_hooks=[lr_decay_hook],
            save_best_so_far_agent=False,
        )


if __name__ == '__main__':
    main()
