import argparse
import sys

import chainer
from chainer import serializers
import gym
import numpy as np

sys.path.append('..')
import random_seed
import env_modifiers
import q_function


def eval_single_run(env, model, phi):
    test_r = 0
    obs = env.reset()
    done = False
    while not done:
        s = chainer.Variable(np.expand_dims(phi(obs), 0))
        qout = model(s)
        a = qout.greedy_actions.data[0]
        obs, r, done, info = env.step(a)
        test_r += r
    return test_r


def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--window-visible', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.set_defaults(render=False)
    args = parser.parse_args()

    random_seed.set_random_seed(args.seed)

    env = gym.make(args.env)
    timestep_limit = env.spec.timestep_limit
    env_modifiers.make_timestep_limited(env, timestep_limit)
    if args.render:
        env_modifiers.make_rendered(env)

    obs_size = np.asarray(env.observation_space.shape).prod()
    action_size = np.asarray(env.action_space.shape).prod()

    q_func = q_function.FCSIContinuousQFunction(
        obs_size, action_size, 100, 2, env.action_space)
    serializers.load_hdf5(args.model, q_func)

    scores = []

    def phi(obs):
        return obs.astype(np.float32)

    for i in range(args.n_runs):
        score = eval_single_run(env, q_func, phi)
        print('Run {}: {}'.format(i, score))
        scores.append(score)
    print('Average: {}'.format(sum(scores) / args.n_runs))


if __name__ == '__main__':
    main()
