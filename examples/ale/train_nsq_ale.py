from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import argparse
import os
import random

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'

from chainer import links as L

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl.agents import nsq
from chainerrl.envs import ale
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers import rmsprop_async
from chainerrl import spaces

from dqn_phi import dqn_phi


def main():

    import logging
    # logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--use-sdl', action='store_true', default=False)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=4 * 10 ** 6)
    parser.add_argument('--outdir', type=str, default='nsq_output')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--eval-interval', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    def make_env(process_idx, test):
        env = ale.ALE(args.rom, use_sdl=args.use_sdl,
                      treat_life_lost_as_terminal=not test)
        if not test:
            misc.env_modifiers.make_reward_clipped(env, -1, 1)
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

    def action_sampler():
        return chainerrl.misc.sample_from_space(action_space)

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
            action_sampler)
        # Suppress the explorer logger
        explorer.logger.setLevel(logging.INFO)
        return nsq.NSQ(q_func, opt, t_max=5, gamma=0.99,
                       i_target=40000,
                       explorer=explorer, phi=dqn_phi)

    if args.demo:
        env = make_env(0, True)
        agent = make_agent(0)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        explorer = explorers.ConstantEpsilonGreedy(0.05, action_sampler)

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
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            eval_explorer=explorer,
            global_step_hooks=[lr_decay_hook])

if __name__ == '__main__':
    main()
