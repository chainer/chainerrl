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
import nstep_q_learning
import ale
import random_seed
import async
import rmsprop_ones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use-sdl', action='store_true')
    parser.set_defaults(use_sdl=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    n_actions = ale.ALE(args.rom).number_of_actions

    def agent_func():
        head = dqn_head.NIPSDQNHead()
        q_func = fc_tail_q_function.FCTailQFunction(
            head, 256, n_actions=n_actions)
        opt = rmsprop_ones.RMSpropOnes(lr=1e-3, eps=1e-4)
        opt.setup(q_func)
        opt.add_hook(chainer.optimizer.GradientClipping(1.0))
        # TODO: epsilon scheduling
        epsilon = random.random()
        return nstep_q_learning.NStepQLearning(q_func, opt, 5, 0.99, epsilon)

    def env_func():
        return ale.ALE(args.rom, use_sdl=args.use_sdl)

    def run_func(agent, env):
        total_r = 0
        episode_r = 0

        for i in xrange(1000000):

            total_r += env.reward
            episode_r += env.reward

            action = agent.act(env.state, env.reward, env.is_terminal)

            if env.is_terminal:
                print 'i:{} episode_r:{}'.format(i, episode_r)
                episode_r = 0
                env.initialize()
            else:
                env.receive_action(action)

        print 'pid:{}, total_r:{}'.format(os.getpid(), total_r)

    async.run_async(args.processes, agent_func, env_func, run_func)


if __name__ == '__main__':
    main()
