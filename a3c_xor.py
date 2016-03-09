import multiprocessing as mp
import os
import argparse
import random

import numpy as np

import chainer
from chainer import optimizers

import policy
import v_function
import a3c
import xor
import random_seed
import async


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.1)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    def agent_func():
        pi = policy.FCSoftmaxPolicy(2, 2, 10, 2)
        v = v_function.FCVFunction(2, 10, 2)
        opt = optimizers.RMSprop(lr=args.lr)
        opt.setup(chainer.ChainList(pi, v))
        return a3c.A3C(pi, v, opt, args.t_max, args.gamma, args.beta)

    def env_func():
        return xor.XOR()

    def run_func(agent, env):
        total_r = 0
        episode_r = 0

        for i in xrange(args.steps):

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
