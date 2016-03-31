import multiprocessing as mp
import os
import argparse
import random
import tempfile
import json
import subprocess
import sys

import numpy as np

import chainer
from chainer import optimizers
from chainer import functions as F

import fc_tail_policy
import fc_tail_v_function
import dqn_head
import policy
import v_function
import a3c
import ale
import random_seed
import async
import rmsprop_ones


def run_func_for_profiling(agent, env):
    # Must be put outside main()  so that cProfile.runctx can see

    total_r = 0
    episode_r = 0

    for i in xrange(1000):

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
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.set_defaults(use_sdl=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    if args.outdir is not None:
        if os.path.exists(args.outdir):
            if not os.path.isdir(args.outdir):
                raise RuntimeError('{} is not a directory'.format(args.outdir))
        else:
            os.makedirs(args.outdir)
        outdir = args.outdir
    else:
        outdir = tempfile.mkdtemp()

    print 'Output files are saved in {}'.format(outdir)

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # Save `git status`
    with open(os.path.join(outdir, 'git-status.txt'), 'w') as f:
        f.write(subprocess.check_output(['git', 'status']))

    # Save `git log`
    with open(os.path.join(outdir, 'git-log.txt'), 'w') as f:
        f.write(subprocess.check_output(['git', 'log']))

    # Save `git diff`
    with open(os.path.join(outdir, 'git-diff.txt'), 'w') as f:
        f.write(subprocess.check_output(['git', 'diff']))

    n_actions = ale.ALE(args.rom).number_of_actions

    def agent_func(process_idx):
        head = dqn_head.NIPSDQNHead()
        pi = fc_tail_policy.FCTailPolicy(
            head, head.n_output_channels, n_actions=n_actions)
        v = fc_tail_v_function.FCTailVFunction(head, head.n_output_channels)
        # opt = optimizers.RMSprop(lr=1e-3)
        opt = rmsprop_ones.RMSpropOnes(lr=1e-3, eps=1e-2, alpha=0.999)
        # opt = rmsprop_ones.RMSpropOnes(lr=1e-4, eps=1e-1)
        # opt = optimizers.RMSpropGraves(
        #     lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)
        model = chainer.ChainList(pi, v)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(2))
        return a3c.A3C(model, opt, args.t_max, 0.99, beta=args.beta, process_idx=process_idx)

    def env_func(process_idx):
        return ale.ALE(args.rom, use_sdl=args.use_sdl)

    def run_func(process_idx, agent, env):
        total_r = 0
        episode_r = 0
        max_score = None

        try:
            for i in xrange(args.steps):

                total_r += env.reward
                episode_r += env.reward

                action = agent.act(env.state, env.reward, env.is_terminal)

                if env.is_terminal:
                    if process_idx == 0:
                        print 'i:{} episode_r:{}'.format(i, episode_r)
                        if max_score == None or episode_r > max_score:
                            if max_score is not None:
                                # Save the best model so far
                                print 'The best score is updated {} -> {}'.format(
                                    max_score, episode_r)
                                filename = os.path.join(
                                    outdir, '{}.h5'.format(i))
                                agent.save_model(filename)
                                print 'Saved the current best model to {}'.format(
                                    filename)
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
                print >> sys.stderr, 'Saved the current model to {}'.format(
                    outdir)
            raise

        if process_idx == 0:
            print '{} pid:{}, total_r:{}'.format(outdir, os.getpid(), total_r)
            # Save the final model
            agent.save_model(
                os.path.join(outdir, '{}_finish.h5'.format(args.steps)))
            print 'Saved the current model to {}'.format(outdir)

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
