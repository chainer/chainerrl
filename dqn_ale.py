import multiprocessing as mp
import os
import argparse
import random

import numpy as np

import chainer
from chainer import optimizers

import fc_tail_q_function
import dqn_head
import q_function
from dqn import DQN
import ale
import random_seed
import async
import rmsprop_ones
import replay_buffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-sdl', action='store_true')
    parser.set_defaults(use_sdl=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    n_actions = ale.ALE(args.rom).number_of_actions

    head = dqn_head.NIPSDQNHead()
    q_func = fc_tail_q_function.FCTailQFunction(
        head, 256, n_actions=n_actions)
    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-4)
    opt.setup(q_func)
    opt.add_hook(chainer.optimizer.GradientClipping(1.0))
    rbuf = replay_buffer.ReplayBuffer(1e6)
    # TODO: epsilon scheduling
    agent = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                epsilon=0.2, replay_start_size=1000, target_update_frequency=1000)

    env = ale.ALE(args.rom, use_sdl=args.use_sdl)

    episode_r = 0

    for i in xrange(1000000):

        episode_r += env.reward

        action = agent.act(env.state, env.reward, env.is_terminal)

        if env.is_terminal:
            print 'i:{} episode_r:{}'.format(i, episode_r)
            episode_r = 0
            env.initialize()
        else:
            env.receive_action(action)

if __name__ == '__main__':
    main()
