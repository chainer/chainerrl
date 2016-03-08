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
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    def agent_func():
        pi = policy.FCSoftmaxPolicy(2, 2, 10, 2)
        v = v_function.FCVFunction(2, 10, 2)
        opt = optimizers.RMSprop()
        opt.setup(chainer.ChainList(pi, v))
        return a3c.A3C(pi, v, opt, 5, 0.9)

    def env_func():
        return xor.XOR()

    def run_func(agent, env):
        total_r = 0

        for i in xrange(2000):
            if env.is_terminal:
                env.initialize()
            action = agent.act(env.state, env.reward, env.is_terminal)
            env.receive_action(action)
            total_r += env.reward

        print 'pid:{}, total_r:{}'.format(os.getpid(), total_r)

    async.run_async(args.processes, agent_func, env_func, run_func)


if __name__ == '__main__':
    main()
