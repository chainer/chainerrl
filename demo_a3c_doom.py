import argparse

import chainer
from chainer import serializers
import numpy as np

import random_seed
import doom_env
from train_a3c_doom import phi, A3CFF, A3CLSTM


def eval_single_run(env, model, phi, deterministic=False):
    model.reset_state()
    test_r = 0
    obs = env.reset()
    done = False
    while not done:
        s = chainer.Variable(np.expand_dims(phi(obs), 0))
        pout = model.pi_and_v(s)[0]
        model.unchain_backward()
        if deterministic:
            a = pout.most_probable_actions[0]
        else:
            a = pout.action_indices[0]
        obs, r, done, info = env.step(a)
        test_r += r
    return test_r


def eval_single_random_run(env):
    test_r = 0
    obs = env.reset()
    done = False
    while not done:
        a = np.random.randint(env.n_actions)
        obs, r, done, info = env.step(a)
        test_r += r
    return test_r


def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sleep', type=float, default=0)
    parser.add_argument('--scenario', type=str, default='basic')
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--window-visible', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.set_defaults(window_visible=False)
    parser.set_defaults(use_lstm=False)
    parser.set_defaults(deterministic=False)
    parser.set_defaults(random=False)
    args = parser.parse_args()

    random_seed.set_random_seed(args.seed)

    n_actions = doom_env.DoomEnv(
        window_visible=False, scenario=args.scenario).n_actions

    if not args.random:
        if args.use_lstm:
            model = A3CLSTM(n_actions)
        else:
            model = A3CFF(n_actions)
        serializers.load_hdf5(args.model, model)

    scores = []
    env = doom_env.DoomEnv(window_visible=args.window_visible,
                           scenario=args.scenario,
                           sleep=args.sleep)
    for i in range(args.n_runs):
        if args.random:
            score = eval_single_random_run(env)
        else:
            score = eval_single_run(
                env, model, phi, deterministic=args.deterministic)
        print('Run {}: {}'.format(i, score))
        scores.append(score)
    print('Average: {}'.format(sum(scores) / args.n_runs))


if __name__ == '__main__':
    main()
