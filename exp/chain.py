from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

import gym
gym.undo_logger_setup()  # NOQA
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np

import sys
sys.path.insert(0, ".")

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DuelingDQN
from chainerrl import replay_buffer

from chain_env import ChainEnv
from grid_env import GridEnv
from myseq import MySequence

def parse_activation(activation_str):
    if activation_str == 'relu':
        return F.relu
    elif activation_str == 'elu':
        return F.elu
    elif activation_str == 'lrelu':
        return F.leaky_relu
    else:
        raise RuntimeError(
            'Not supported activation: {}'.format(activation_str))


def parse_arch(arch, n_actions, activation):
    if arch == 'nature':
        return links.Sequence(
            links.NatureDQNHead(activation=activation),
            L.Linear(512, n_actions),
            DiscreteActionValue)
    elif arch == 'nips':
        return links.Sequence(
            links.NIPSDQNHead(activation=activation),
            L.Linear(256, n_actions),
            DiscreteActionValue)
    elif arch == 'dueling':
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))


def parse_agent(agent):
    return {'DQN': agents.DQN,
            'DoubleDQN': agents.DoubleDQN,
            'SARSA': agents.SARSA,
            'PAL': agents.PAL}[agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--use-sdl', action='store_true', default=False)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=1000)
    parser.add_argument('--final-epsilon', type=float, default=0.1)
    parser.add_argument('--eval-epsilon', type=float, default=0.05)
    parser.add_argument('--arch', type=str, default='nature',
                        choices=['nature', 'nips', 'dueling'])
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--buffer-size', type=int, default=1000)
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--max-episode-len', type=int,
                        default=5 * 60 * 60 // 4,  # 5 minutes with 60/4 fps
                        help='Maximum number of steps for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=100)
    parser.add_argument('--target-update-interval',
                        type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--agent', type=str, default='DQN',
                        choices=['DQN', 'DoubleDQN', 'PAL', 'SARSA'])
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--noise-constant', type=float, default=-1)
    parser.add_argument('--prop', action='store_true', default=False)
    parser.add_argument('--adam', action='store_true', default=True)
    parser.add_argument('--orig-noise', action='store_true', default=False)
    parser.add_argument('--last-noise', type=int, default=0)
    parser.add_argument('--entropy-coef', type=float, default=0)
    parser.add_argument('--noise-coef', type=float, default=1)
    parser.add_argument('--init-method', type=str, default='/out')

    parser.add_argument('--noisy-y', action='store_true', default=False)
    parser.add_argument('--noisy-t', action='store_true', default=False)
    parser.add_argument('--save-img', action='store_true', default=False)

    parser.add_argument('--env', type=str, default='grid')

    parser.add_argument('--len', type=int, default=20)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    chain_len = args.len

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        if args.env == "chain":
            env = ChainEnv(chain_len)
        elif args.env == "grid":
            env = GridEnv(args.outdir, chain_len, save_img=args.save_img,)
        elif args.env == "car":
            env = gym.make("MountainCar-v0")
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    try:
        n_obs = env.observation_space.n
    except:
        n_obs = env.observation_space.shape[0]
    activation = parse_activation(args.activation)
    q_func = MySequence(n_obs, n_actions)

    """
    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((n_obs), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'diagram'))
    """

    if args.adam:
        opt = optimizers.Adam(args.lr)
    else:
        # Use the same hyper parameters as the Nature paper's
        opt = optimizers.RMSpropGraves(
            lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)

    rbuf = replay_buffer.ReplayBuffer(args.buffer_size)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    entropy = None
    if args.noisy_net_sigma is not None and args.noisy_net_sigma > 0:
        entropy = links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma, constant=args.noise_constant,
            prev=args.orig_noise, noise_coef=args.noise_coef, init_method=args.init_method)
        # Turn off explorer
        explorer = explorers.Greedy()

        if args.last_noise > 0:
            for e in entropy[:-args.last_noise]:
                e.off = True

    """
    print(n_obs)
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((n_obs), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'diagram2'))
    """

    opt.setup(q_func)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32)

    Agent = parse_agent(args.agent)
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.9,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=args.target_update_interval,
                  clip_delta=args.clip_delta,
                  update_interval=args.update_interval,
                  batch_accumulator='sum',
                  minibatch_size=args.minibatch_size,
                  phi=phi, entropy=entropy, entropy_coef=args.entropy_coef,
                  vis=env, noisy_y=args.noisy_y, noisy_t=args.noisy_t,
                  plot=args.save_img)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        # In testing DQN, randomly select 5% of actions
        eval_explorer = explorers.Greedy()
        #explorers.ConstantEpsilonGreedy(
        #    args.eval_epsilon, lambda: np.random.randint(n_actions))
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_explorer=eval_explorer,
            max_episode_len=args.max_episode_len,
            eval_env=eval_env,
            save_best_so_far_agent=False,
        )


if __name__ == '__main__':
    main()
