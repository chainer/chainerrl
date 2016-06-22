from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import argparse
import sys

import chainer
from chainer import serializers
import gym
import numpy as np

sys.path.append('..')
import random_seed
from train_a3c_continuous import phi, A3CLSTMGaussian
import env_modifiers


def eval_single_run(env, model, phi):
    model.reset_state()
    test_r = 0
    obs = env.reset()
    done = False
    while not done:
        s = chainer.Variable(np.expand_dims(phi(obs), 0))
        pout = model.pi_and_v(s)[0]
        model.unchain_backward()
        a = pout.sampled_actions.data[0]
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

    model = A3CLSTMGaussian(obs_size, action_size)
    serializers.load_hdf5(args.model, model)

    scores = []

    for i in range(args.n_runs):
        score = eval_single_run(env, model, phi)
        print('Run {}: {}'.format(i, score))
        scores.append(score)
    print('Average: {}'.format(sum(scores) / args.n_runs))


if __name__ == '__main__':
    main()
