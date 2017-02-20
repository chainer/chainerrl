from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse

import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
import numpy as np

from chainerrl.action_value import DiscreteActionValue
from chainerrl.agents import acer
from chainerrl.distribution import SoftmaxDistribution
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl.replay_buffer import EpisodicReplayBuffer


def phi(obs):
    return obs.astype(np.float32)


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--n-times-replay', type=int, default=8)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()

    logging.getLogger().setLevel(args.logger_level)

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        if args.monitor and process_idx == 0:
            env = gym.wrappers.Monitor(env, args.outdir)
        # Scale rewards observed by agents
        if not test:
            misc.env_modifiers.make_reward_filtered(
                env, lambda x: x * args.reward_scale_factor)
        if args.render and process_idx == 0 and not test:
            misc.env_modifiers.make_rendered(env)
        return env

    sample_env = gym.make(args.env)
    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    n_hidden_channels = 200

    model = acer.ACERSeparateModel(
        pi=links.Sequence(
            L.Linear(obs_space.low.size, n_hidden_channels),
            F.relu,
            L.Linear(n_hidden_channels, action_space.n, wscale=1e-3),
            SoftmaxDistribution),
        q=links.Sequence(
            L.Linear(obs_space.low.size, n_hidden_channels),
            F.relu,
            L.Linear(n_hidden_channels, action_space.n, wscale=1e-3),
            DiscreteActionValue),
    )

    opt = rmsprop_async.RMSpropAsync(
        lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    replay_buffer = EpisodicReplayBuffer(10 ** 5 // args.processes)
    agent = acer.DiscreteACER(model, opt, t_max=args.t_max, gamma=0.99,
                              replay_buffer=replay_buffer,
                              n_times_replay=args.n_times_replay,
                              beta=args.beta, phi=phi)
    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(0, True)
        mean, median, stdev = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev'.format(
            args.eval_n_runs, mean, median, stdev))
    else:
        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_frequency=args.eval_frequency,
            max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
