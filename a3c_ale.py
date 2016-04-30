import os
import argparse
import sys

import numpy as np

import chainer
from chainer import optimizers
from chainer import functions as F

import policy
import v_function
import dqn_head
import a3c
import ale
import random_seed
import async
import rmsprop_ones
from prepare_output_dir import prepare_output_dir


def run_func_for_profiling(agent, env):
    # Must be put outside main()  so that cProfile.runctx can see

    total_r = 0
    episode_r = 0

    for i in range(1000):

        total_r += env.reward
        episode_r += env.reward

        action = agent.act(env.state, env.reward, env.is_terminal)

        if env.is_terminal:
            print('i:{} episode_r:{}'.format(i, episode_r))
            episode_r = 0
            env.initialize()
        else:
            env.receive_action(action)

    print('pid:{}, total_r:{}'.format(os.getpid(), total_r))


def phi(screens):
    assert len(screens) == 4
    assert screens[0].dtype == np.uint8
    raw_values = np.asarray(screens, dtype=np.float32)
    # [0,255] -> [-128, 127]
    raw_values -= 128
    # [-128, 127] -> [-1, 1)
    raw_values /= 128.0
    return raw_values


def main():

    # This line makes execution much faster, I don't know why
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--use-sdl', action='store_true')
    parser.add_argument('--t-max', type=int, default=20)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.set_defaults(use_sdl=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    def agent_func(process_idx):
        head = dqn_head.NIPSDQNHead()
        pi = policy.FCSoftmaxPolicy(head.n_output_channels, n_actions)
        v = v_function.FCVFunction(head.n_output_channels)
        model = chainer.ChainList(head, pi, v)

        # Initialize last layers with uniform random values following:
        # http://arxiv.org/abs/1509.02971
        for param in pi[-1].params():
            param.data[:] = \
                np.random.uniform(-3e-3, 3e-3, size=param.data.shape)
        for param in v[-1].params():
            param.data[:] = \
                np.random.uniform(-3e-4, 3e-4, size=param.data.shape)

        def pv_func(model, state):
            head, pi, v = model
            out = head(state)
            return pi(out), v(out)

        # opt = optimizers.RMSprop(lr=1e-3)
        opt = rmsprop_ones.RMSpropOnes(lr=7e-4, eps=1e-2, alpha=0.99)
        # opt = rmsprop_ones.RMSpropOnes(lr=1e-4, eps=1e-1)
        # opt = optimizers.RMSpropGraves(
        #     lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        return a3c.A3C(model, pv_func, opt, args.t_max, 0.99, beta=args.beta,
                       process_idx=process_idx, phi=phi)

    def env_func(process_idx):
        return ale.ALE(args.rom, use_sdl=args.use_sdl)

    def run_func(process_idx, agent, env):
        total_r = 0
        episode_r = 0
        max_score = None

        try:
            for i in range(args.steps):

                agent.optimizer.lr = (args.steps - i) / args.steps * args.lr

                total_r += env.reward
                episode_r += env.reward

                action = agent.act(env.state, env.reward, env.is_terminal)

                if env.is_terminal:
                    if process_idx == 0:
                        print('{} i:{} lr:{} episode_r:{}'.format(
                            outdir, i, agent.optimizer.lr, episode_r))
                        with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
                            print(i, episode_r, file=f)
                        if max_score == None or episode_r > max_score:
                            if max_score is not None:
                                # Save the best model so far
                                print('The best score is updated {} -> {}'.format(
                                    max_score, episode_r))
                                filename = os.path.join(
                                    outdir, '{}.h5'.format(i))
                                agent.save_model(filename)
                                print('Saved the current best model to {}'.format(
                                    filename))
                            max_score = episode_r
                    episode_r = 0
                    env.initialize()
                else:
                    env.receive_action(action)
        except KeyboardInterrupt:
            if process_idx == 0:
                # Save the current model before being killed
                agent.save_model(os.path.join(
                    outdir, '{}_keyboardinterrupt.h5'.format(i)))
                print('Saved the current model to {}'.format(
                    outdir), file=sys.stderr)
            raise

        if process_idx == 0:
            print('{} pid:{}, total_r:{}'.format(outdir, os.getpid(), total_r))
            # Save the final model
            agent.save_model(
                os.path.join(outdir, '{}_finish.h5'.format(args.steps)))
            print('Saved the current model to {}'.format(outdir))

    if args.profile:

        def profile_run_func(process_idx, agent, env):
            import cProfile
            cProfile.runctx('run_func_for_profiling(agent, env)',
                            globals(), locals(),
                            'profile-{}.out'.format(os.getpid()))

        async.run_async(args.processes, agent_func, env_func, profile_run_func)
    else:
        async.run_async(args.processes, agent_func, env_func, run_func)


if __name__ == '__main__':
    main()
