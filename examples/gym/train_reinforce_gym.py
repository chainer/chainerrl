"""An example of training a REINFORCE agent against OpenAI Gym envs.

This script is an example of training a REINFORCE agent against OpenAI Gym
envs. Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_reinforce_gym.py

To solve InvertedPendulum-v1, run:
    python train_reinforce_gym.py --env InvertedPendulum-v1
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse
import os

import chainer
import gym
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl import experiments
from chainerrl import misc


def phi(obs):
    return obs.astype(np.float32)


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--beta', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--steps', type=int, default=10 ** 5)
    parser.add_argument('--eval-interval', type=int, default=10 ** 4)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()

    logging.getLogger().setLevel(args.logger_level)

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(test):
        env = gym.make(args.env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        # Scale rewards observed by agents
        if not test:
            misc.env_modifiers.make_reward_filtered(
                env, lambda x: x * args.reward_scale_factor)
        if args.render and not test:
            misc.env_modifiers.make_rendered(env)
        return env

    train_env = make_env(test=False)
    timestep_limit = train_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = train_env.observation_space
    action_space = train_env.action_space

    # Switch policy types accordingly to action space types
    if isinstance(action_space, gym.spaces.Box):
        model = chainerrl.policies.FCGaussianPolicyWithFixedCovariance(
            obs_space.low.size,
            action_space.low.size,
            var=0.1,
            n_hidden_channels=200,
            n_hidden_layers=2,
            nonlinearity=chainer.functions.leaky_relu,
        )
    else:
        model = chainerrl.policies.FCSoftmaxPolicy(
            obs_space.low.size,
            action_space.n,
            n_hidden_channels=200,
            n_hidden_layers=2,
            nonlinearity=chainer.functions.leaky_relu,
        )

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [model(np.zeros_like(obs_space.low, dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    opt = chainer.optimizers.Adam(alpha=args.lr)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(1))

    agent = chainerrl.agents.REINFORCE(
        model, opt, beta=args.beta, phi=phi, batchsize=args.batchsize)
    if args.load:
        agent.load(args.load)

    eval_env = make_env(test=True)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=train_env,
            eval_env=eval_env,
            outdir=args.outdir,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
