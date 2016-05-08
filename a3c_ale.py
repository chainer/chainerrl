import argparse
import multiprocessing as mp
import os
import sys
import statistics
import time

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
import rmsprop_async
from prepare_output_dir import prepare_output_dir
from nonbias_weight_decay import NonbiasWeightDecay


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


def eval_performance(rom, p_func, n_runs):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []
    for i in range(n_runs):
        env = ale.ALE(rom, treat_life_lost_as_terminal=False)
        test_r = 0
        while not env.is_terminal:
            s = chainer.Variable(np.expand_dims(phi(env.state), 0))
            pout = p_func(s)
            a = pout.action_indices[0]
            test_r += env.receive_action(a)
        scores.append(test_r)
        print('test_{}:'.format(i), test_r)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


def main():

    # Prevent numpy from using multiple threads
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
    parser.add_argument('--steps', type=int, default=10 ** 8)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.set_defaults(use_sdl=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    def pv_func(model, state):
        head, pi, v = model
        out = head(state)
        return pi(out), v(out)

    def model_opt():
        head = dqn_head.NIPSDQNHead()
        pi = policy.FCSoftmaxPolicy(head.n_output_channels, n_actions)
        v = v_function.FCVFunction(head.n_output_channels)
        # Initialize last layers with uniform random values following:
        # http://arxiv.org/abs/1509.02971
        for param in pi[-1].params():
            param.data[:] = \
                np.random.uniform(-3e-3, 3e-3, size=param.data.shape)
        for param in v[-1].params():
            param.data[:] = \
                np.random.uniform(-3e-4, 3e-4, size=param.data.shape)
        model = chainer.ChainList(head, pi, v)
        opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))
        return model, opt

    model, opt = model_opt()

    shared_params = async.share_params_as_shared_arrays(model)
    shared_states = async.share_states_as_shared_arrays(opt)

    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    # Write a header line first
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    def run_func(process_idx):
        total_r = 0
        episode_r = 0

        env = ale.ALE(args.rom, use_sdl=args.use_sdl)
        model, opt = model_opt()
        async.set_shared_params(model, shared_params)
        async.set_shared_states(opt, shared_states)

        agent = a3c.A3C(model, pv_func, opt, args.t_max, 0.99, beta=args.beta,
                        process_idx=process_idx, phi=phi)

        try:

            global_t = 0
            local_t = 0

            while True:

                # Get and increment the global counter
                with counter.get_lock():
                    counter.value += 1
                    global_t = counter.value
                local_t += 1

                if global_t > args.steps:
                    break

                agent.optimizer.lr = (
                    args.steps - global_t - 1) / args.steps * args.lr

                total_r += env.reward
                episode_r += env.reward

                action = agent.act(env.state, env.reward, env.is_terminal)

                if env.is_terminal:
                    if process_idx == 0:
                        print('{} global_t:{} local_t:{} lr:{} episode_r:{}'.format(
                            outdir, global_t, local_t, agent.optimizer.lr, episode_r))
                    episode_r = 0
                    env.initialize()
                else:
                    env.receive_action(action)

                if global_t % args.eval_frequency == 0:
                    # Evaluation
                    def p_func(s):
                        pout, _ = agent.pv_func(agent.model, s)
                        return pout
                    mean, median, stdev = eval_performance(
                        args.rom, p_func, args.eval_n_runs)
                    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
                        elapsed = time.time() - start_time
                        record = (global_t, elapsed, mean, median, stdev)
                        print('\t'.join(str(x) for x in record), file=f)
                    with max_score.get_lock():
                        if mean > max_score.value:
                            # Save the best model so far
                            print('The best score is updated {} -> {}'.format(
                                max_score.value, mean))
                            filename = os.path.join(
                                outdir, '{}.h5'.format(global_t))
                            agent.save_model(filename)
                            print('Saved the current best model to {}'.format(
                                filename))
                            max_score.value = mean

        except KeyboardInterrupt:
            if process_idx == 0:
                # Save the current model before being killed
                agent.save_model(os.path.join(
                    outdir, '{}_keyboardinterrupt.h5'.format(global_t)))
                print('Saved the current model to {}'.format(
                    outdir), file=sys.stderr)
            raise

        if global_t == args.steps + 1:
            # Save the final model
            agent.save_model(
                os.path.join(outdir, '{}_finish.h5'.format(args.steps)))
            print('Saved the final model to {}'.format(outdir))

    if args.profile:

        def profile_run_func(process_idx):
            import cProfile
            cProfile.runctx('run_func_for_profiling()',
                            globals(), locals(),
                            'profile-{}.out'.format(os.getpid()))

        async.run_async(args.processes, profile_run_func)
    else:
        async.run_async(args.processes, run_func)


if __name__ == '__main__':
    main()
