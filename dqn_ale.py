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
import ale
import random_seed
import async
import rmsprop_ones
import replay_buffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-sdl', action='store_true')
    parser.set_defaults(use_sdl=False)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=int(1e6))
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--arch', type=str, default='nature')
    parser.add_argument('--steps', type=int, default=int(1e7))
    parser.add_argument('--replay-start-size', type=int, default=int(5e4))
    parser.add_argument('--target-update-frequency',
                        type=int, default=int(1e4))
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    n_actions = ale.ALE(args.rom).number_of_actions

    if args.arch == 'nature':
        head = dqn_head.NatureDQNHead()
        q_func = fc_tail_q_function.FCTailQFunction(
            head, 512, n_actions=n_actions)
    elif args.arch == 'nips':
        head = dqn_head.NIPSDQNHead()
        q_func = fc_tail_q_function.FCTailQFunction(
            head, 256, n_actions=n_actions)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(args.arch))

    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-4)
    opt.setup(q_func)
    # opt.add_hook(chainer.optimizer.GradientClipping(1.0))
    rbuf = replay_buffer.ReplayBuffer(1e5)
    agent = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                epsilon=1.0, replay_start_size=args.replay_start_size,
                target_update_frequency=args.target_update_frequency)

    if len(args.model) > 0:
        agent.load_model(args.model)

    env = ale.ALE(args.rom, use_sdl=args.use_sdl)

    episode_r = 0

    for i in xrange(args.steps):
        try:
            episode_r += env.reward

            action = agent.act(env.state, env.reward, env.is_terminal)

            if agent.epsilon >= 0.1:
                agent.epsilon -= 0.9 / args.final_exploration_frames

            if env.is_terminal:
                print 'i:{} epsilon:{} episode_r:{}'.format(i, agent.epsilon, episode_r)
                episode_r = 0
                env.initialize()
            else:
                env.receive_action(action)
        except KeyboardInterrupt:
            # Save the current model before being killed
            tempdir = tempfile.mkdtemp(prefix='drill')
            agent.save_model(tempdir + '/{}_keyboardinterrupt.h5'.format(i))
            print >> sys.stderr, 'Saved the current model to {}'.format(
                tempdir)
            raise

    # Save the final model
    tempdir = tempfile.mkdtemp(prefix='drill')
    agent.save_model(tempdir + '/{}_finish.h5'.format(args.steps))
    print 'Saved the current model to {}'.format(tempdir)

if __name__ == '__main__':
    main()
