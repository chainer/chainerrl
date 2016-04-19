import argparse
import os
import sys

import chainer
from chainer import optimizers
from chainer import functions as F

import fc_tail_q_function
import dqn_head
from dqn import DQN
import ale
import random_seed
import replay_buffer
from prepare_output_dir import prepare_output_dir
import oplu


def parse_activation(activation_str):
    if activation_str == 'relu':
        return F.relu
    elif activation_str == 'elu':
        return F.elu
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
        head = dqn_head.NatureDQNHeadCReLU()
        return fc_tail_q_function.FCTailQFunction(
            head, 512, n_actions=n_actions)
    elif arch == 'nips':
        head = dqn_head.NIPSDQNHead(activation=activation)
        return fc_tail_q_function.FCTailQFunction(
            head, 256, n_actions=n_actions)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))


def eval_performance(rom, q_func, gpu):
    env = ale.ALE(rom)
    test_r = 0
    while not env.is_terminal:
        s = env.state.reshape((1,) + env.state.shape)
        if gpu >= 0:
            s = chainer.cuda.to_gpu(s)
        a = q_func.sample_epsilon_greedily_with_value(s, 5e-2)[0][0]
        test_r += env.receive_action(a)
    print 'test_r:', test_r
    return test_r


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
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    outdir = prepare_output_dir(args, args.outdir)

    print 'Output files are saved in {}'.format(outdir)

    n_actions = ale.ALE(args.rom).number_of_actions
    activation = parse_activation(args.activation)
    q_func = parse_arch(args.arch, n_actions, activation)

    # Use the same hyper parameters as the Nature paper's
    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)

    opt.setup(q_func)
    # opt.add_hook(chainer.optimizer.GradientClipping(1.0))

    # 10^6 in the Nature paper, but use 10^5 for avoiding out-of-memory
    rbuf = replay_buffer.ReplayBuffer(1e5)

    agent = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                epsilon=1.0, replay_start_size=args.replay_start_size,
                target_update_frequency=args.target_update_frequency,
                clip_delta=args.clip_delta,
                update_frequency=args.update_frequency)

    if len(args.model) > 0:
        agent.load_model(args.model)

    env = ale.ALE(args.rom, use_sdl=args.use_sdl)

    episode_r = 0

    episode_idx = 0
    max_score = None

    for i in xrange(args.steps):

        try:
            if i % (args.steps / 100) == 0:
                # Test performance
                score = eval_performance(args.rom, agent.q_function, args.gpu)
                with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
                    print >> f, i, score
                if max_score is None or score > max_score:
                    if max_score is not None:
                        # Save the best model so far
                        print 'The best score is updated {} -> {}'.format(
                            max_score, score)
                        filename = os.path.join(
                            outdir, '{}_keyboardinterrupt.h5'.format(i))
                        agent.save_model(filename)
                        print 'Saved the current best model to {}'.format(filename)
                    max_score = score

            episode_r += env.reward

            action = agent.act(env.state, env.reward, env.is_terminal)

            if agent.epsilon >= 0.1:
                agent.epsilon -= 0.9 / args.final_exploration_frames

            if env.is_terminal:
                print '{} i:{} episode_idx:{} epsilon:{} episode_r:{}'.format(outdir, i, episode_idx, agent.epsilon, episode_r)
                episode_r = 0
                episode_idx += 1
                env.initialize()
            else:
                env.receive_action(action)
        except KeyboardInterrupt:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                outdir, '{}_keyboardinterrupt.h5'.format(i)))
            print >> sys.stderr, 'Saved the current model to {}'.format(
                outdir)
            raise

    # Save the final model
    agent.save_model(os.path.join(outdir, '{}_finish.h5'.format(args.steps)))
    print 'Saved the current model to {}'.format(outdir)

if __name__ == '__main__':
    main()
