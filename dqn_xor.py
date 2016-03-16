import multiprocessing as mp
import os
import argparse
import random
import sys
import tempfile

import numpy as np

import chainer
from chainer import optimizers

import fc_tail_q_function
import dqn_head
import q_function
from dqn import DQN
import xor
import delayed_xor
import random_seed
import async
import rmsprop_ones
import replay_buffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-frames', type=int, default=100)
    parser.add_argument('--model', type=str, default='')
    parser.set_defaults(use_sdl=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    q_func = q_function.FCSIQFunction(3, 2, 10, 2)

    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-4)
    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(1e5)
    agent = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=0.9, epsilon=0.1,
                replay_start_size=1000, target_update_frequency=1000)

    if len(args.model) > 0:
        agent.load_model(args.model)

    # env = xor.XOR()
    env = delayed_xor.DelayedXOR(5)

    total_r = 0
    episode_r = 0

    for i in xrange(20000):
        try:
            episode_r += env.reward
            total_r += env.reward

            action = agent.act(env.state, env.reward, env.is_terminal)

            if env.is_terminal:
                print 'i:{} epsilon:{} episode_r:{}'.format(i, agent.epsilon, episode_r)
                episode_r = 0
                env.initialize()
            else:
                env.receive_action(action)
        except KeyboardInterrupt:
            tempdir = tempfile.mkdtemp(prefix='drill')
            agent.save_model(tempdir + '/{}_keyboardinterrupt.h5'.format(i))
            print >> sys.stderr, 'Saved the current model to {}'.format(
                tempdir)
            raise

    print 'total_r:{}'.format(total_r)

if __name__ == '__main__':
    main()
