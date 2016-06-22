from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from builtins import open
from builtins import str
from future import standard_library
standard_library.install_aliases()
import argparse
import os
import statistics
import sys
import time

import chainer
from chainer import optimizers
from chainer import functions as F
import numpy as np

sys.path.append('..')
from links import fc_tail_q_function
from links import dqn_head
from links import dqn_head_crelu
from agents.dqn import DQN
from envs import ale
import random_seed
import replay_buffer
from prepare_output_dir import prepare_output_dir
from functions import oplu
from init_like_torch import init_like_torch
from dqn_phi import dqn_phi
from explorers.epsilon_greedy import LinearDecayEpsilonGreedy


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
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))


def eval_performance(rom, q_func, n_runs, gpu):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []
    for i in range(n_runs):
        env = ale.ALE(rom, treat_life_lost_as_terminal=False)
        test_r = 0
        while not env.is_terminal:
            s = np.expand_dims(dqn_phi(env.state), 0)
            if gpu >= 0:
                s = chainer.cuda.to_gpu(s)
            qout = q_func(chainer.Variable(s))
            a = qout.sample_epsilon_greedy_actions(5e-2).data[0]
            test_r += env.receive_action(a)
        scores.append(test_r)
        print('test_{}:'.format(i), test_r)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


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
                        choices=['nature', 'nips', 'nature_crelu'])
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-frequency',
                        type=int, default=10 ** 4)
    parser.add_argument('--update-frequency', type=int, default=1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    args.outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = ale.ALE(args.rom).number_of_actions
    activation = parse_activation(args.activation)
    q_func = parse_arch(args.arch, n_actions, activation)
    init_like_torch(q_func)

    # Use the same hyper parameters as the Nature paper's
    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)

    opt.setup(q_func)
    # opt.add_hook(chainer.optimizer.GradientClipping(1.0))

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = LinearDecayEpsilonGreedy(1.0, 0.1, args.final_exploration_frames,
                                        lambda: np.random.randint(n_actions))
    agent = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                explorer=explorer, replay_start_size=args.replay_start_size,
                target_update_frequency=args.target_update_frequency,
                clip_delta=args.clip_delta,
                update_frequency=args.update_frequency,
                phi=dqn_phi)

    if len(args.model) > 0:
        agent.load_model(args.model)

    env = ale.ALE(args.rom, use_sdl=args.use_sdl)

    episode_r = 0

    episode_idx = 0
    max_score = np.finfo(np.float32).min

    # Write a header line first
    with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    start_time = time.time()

    for i in range(args.steps):

        try:
            if i % (args.steps / 100) == 0:
                # Test performance
                mean, median, stdev = eval_performance(
                    args.rom, agent.q_function, args.eval_n_runs, args.gpu)
                with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
                    elapsed = time.time() - start_time
                    record = (i, elapsed, mean, median, stdev)
                    print('\t'.join(str(x) for x in record), file=f)
                if mean > max_score:
                    if max_score is not None:
                        # Save the best model so far
                        print('The best score is updated {} -> {}'.format(
                            max_score, mean))
                        filename = os.path.join(
                            args.outdir, '{}_keyboardinterrupt.h5'.format(i))
                        agent.save_model(filename)
                        print('Saved the current best model to {}'.format(
                            filename))
                    max_score = mean

            episode_r += env.reward

            action = agent.act(env.state, env.reward, env.is_terminal)

            if env.is_terminal:
                print('{} i:{} episode_idx:{} explorer:{} episode_r:{}'.format(
                    args.outdir, i, episode_idx, agent.explorer, episode_r))
                episode_r = 0
                episode_idx += 1
                env.initialize()
            else:
                env.receive_action(action)
        except KeyboardInterrupt:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                args.outdir, '{}_keyboardinterrupt.h5'.format(i)))
            print('Saved the current model to {}'.format(
                args.outdir), file=sys.stderr)
            raise

    # Save the final model
    agent.save_model(os.path.join(
        args.outdir, '{}_finish.h5'.format(args.steps)))
    print('Saved the current model to {}'.format(args.outdir))

if __name__ == '__main__':
    main()
