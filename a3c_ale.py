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
from chainer import links as L

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
    # [0,255] -> [0, 1]
    raw_values /= 255.0
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


def init_like_torch(link):
    # Mimic torch's default parameter initialization
    # TODO(muupan): Use chainer's initializers when it is merged
    for l in link.links():
        if isinstance(l, L.Linear):
            out_channels, in_channels = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)
            l.b.data[:] = np.random.uniform(-stdv, stdv, size=l.b.data.shape)
        elif isinstance(l, L.Convolution2D):
            out_channels, in_channels, kh, kw = l.W.data.shape
            stdv = 1 / np.sqrt(in_channels * kh * kw)
            l.W.data[:] = np.random.uniform(-stdv, stdv, size=l.W.data.shape)
            l.b.data[:] = np.random.uniform(-stdv, stdv, size=l.b.data.shape)


def train_loop(process_idx, counter, max_score, args, agent, env, start_time):
    try:

        total_r = 0
        episode_r = 0
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
                        args.outdir, global_t, local_t, agent.optimizer.lr, episode_r))
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
                with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
                    elapsed = time.time() - start_time
                    record = (global_t, elapsed, mean, median, stdev)
                    print('\t'.join(str(x) for x in record), file=f)
                with max_score.get_lock():
                    if mean > max_score.value:
                        # Save the best model so far
                        print('The best score is updated {} -> {}'.format(
                            max_score.value, mean))
                        filename = os.path.join(
                            args.outdir, '{}.h5'.format(global_t))
                        agent.save_model(filename)
                        print('Saved the current best model to {}'.format(
                            filename))
                        max_score.value = mean

    except KeyboardInterrupt:
        if process_idx == 0:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                args.outdir, '{}_keyboardinterrupt.h5'.format(global_t)))
            print('Saved the current model to {}'.format(
                args.outdir), file=sys.stderr)
        raise

    if global_t == args.steps + 1:
        # Save the final model
        agent.save_model(
            os.path.join(args.outdir, '{}_finish.h5'.format(args.steps)))
        print('Saved the final model to {}'.format(args.outdir))


def train_loop_with_profile(process_idx, counter, max_score, args, agent, env,
                            start_time):
    import cProfile
    cmd = 'train_loop(process_idx, counter, max_score, args, agent, env, ' \
        'start_time)'
    cProfile.runctx(cmd, globals(), locals(),
                    'profile-{}.out'.format(os.getpid()))


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
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.set_defaults(use_sdl=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    args.outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    def pv_func(model, state):
        head, pi, v = model
        out = head(state)
        return pi(out), v(out)

    def model_opt():
        head = dqn_head.NIPSDQNHead()
        pi = policy.FCSoftmaxPolicy(head.n_output_channels, n_actions)
        v = v_function.FCVFunction(head.n_output_channels)
        model = chainer.ChainList(head, pi, v)
        init_like_torch(model)
        opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        if args.weight_decay > 0:
            opt.add_hook(NonbiasWeightDecay(args.weight_decay))
        return model, opt

    model, opt = model_opt()

    shared_params = async.share_params_as_shared_arrays(model)
    shared_states = async.share_states_as_shared_arrays(opt)

    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    # Write a header line first
    with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    def run_func(process_idx):
        env = ale.ALE(args.rom, use_sdl=args.use_sdl)
        model, opt = model_opt()
        async.set_shared_params(model, shared_params)
        async.set_shared_states(opt, shared_states)

        agent = a3c.A3C(model, pv_func, opt, args.t_max, 0.99, beta=args.beta,
                        process_idx=process_idx, phi=phi)

        if args.profile:
            train_loop_with_profile(process_idx, counter, max_score,
                                    args, agent, env, start_time)
        else:
            train_loop(process_idx, counter, max_score,
                       args, agent, env, start_time)

    async.run_async(args.processes, run_func)


if __name__ == '__main__':
    main()
