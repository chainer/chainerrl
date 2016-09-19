from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import str
from future import standard_library
standard_library.install_aliases()
import argparse
import sys

from chainer import optimizers
from chainer import functions as F
import numpy as np

sys.path.append('..')
from links import fc_tail_q_function
from links import dqn_head
from links import dqn_head_crelu
from links.dueling_dqn import DuelingDQN
from agents.dqn import DQN
from agents.double_dqn import DoubleDQN
from agents.pal import PAL
from envs import ale
import random_seed
import replay_buffer
from prepare_output_dir import prepare_output_dir
from functions import oplu
from init_like_torch import init_like_torch
from dqn_phi import dqn_phi
from explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from explorers.epsilon_greedy import ConstantEpsilonGreedy
import run_dqn


def parse_activation(activation_str):
    if activation_str == 'relu':
        return F.relu
    elif activation_str == 'elu':
        return F.elu
    elif activation_str == 'oplu':
        return oplu.oplu
    elif activation_str == 'lrelu':
        return F.leaky_relu
    elif activation_str == 'oplu':
        return oplu.oplu
    else:
        raise RuntimeError(
            'Not supported activation: {}'.format(activation_str))


def parse_arch(arch, n_actions, activation):
    if arch == 'nature':
        head = dqn_head.NatureDQNHead(activation=activation)
        return fc_tail_q_function.FCTailQFunction(
            head, 512, n_actions=n_actions)
    if arch == 'nature_crelu':
        head = dqn_head_crelu.NatureDQNHeadCReLU()
        return fc_tail_q_function.FCTailQFunction(
            head, 512, n_actions=n_actions)
    elif arch == 'nips':
        head = dqn_head.NIPSDQNHead(activation=activation)
        return fc_tail_q_function.FCTailQFunction(
            head, 256, n_actions=n_actions)
    elif arch == 'dueling':
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))


def parse_agent(agent):
    return {'DQN': DQN, 'DoubleDQN': DoubleDQN, 'PAL': PAL}[agent]


def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('rom', type=str)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-sdl', action='store_true')
    parser.set_defaults(use_sdl=False)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--arch', type=str, default='nature',
                        choices=['nature', 'nips', 'nature_crelu', 'dueling'])
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-frequency',
                        type=int, default=10 ** 4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 5)
    parser.add_argument('--update-frequency', type=int, default=4)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--agent', type=str, default='DQN',
                        choices=['DQN', 'DoubleDQN', 'PAL'])
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    env = ale.ALE(args.rom, use_sdl=args.use_sdl)
    eval_env = ale.ALE(args.rom, use_sdl=args.use_sdl,
                       treat_life_lost_as_terminal=False)

    args.outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = env.number_of_actions
    activation = parse_activation(args.activation)
    q_func = parse_arch(args.arch, n_actions, activation)
    init_like_torch(q_func)

    # Use the same hyper parameters as the Nature paper's
    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)
    # opt.add_hook(chainer.optimizer.GradientClipping(1.0))

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = LinearDecayEpsilonGreedy(1.0, 0.1,
                                        args.final_exploration_frames,
                                        lambda: np.random.randint(n_actions))
    Agent = parse_agent(args.agent)
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_frequency=args.target_update_frequency,
                  clip_delta=args.clip_delta,
                  update_frequency=args.update_frequency,
                  batch_accumulator='sum', phi=dqn_phi)

    if len(args.model) > 0:
        agent.load_model(args.model)

    eval_explorer = ConstantEpsilonGreedy(
        5e-2, lambda: np.random.randint(n_actions))
    run_dqn.run_dqn(
        agent=agent, env=env, steps=args.steps,
        eval_n_runs=args.eval_n_runs, eval_frequency=args.eval_frequency,
        outdir=args.outdir, eval_explorer=eval_explorer,
        eval_env=eval_env)

if __name__ == '__main__':
    main()
