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
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl import q_functions
from chainerrl.replay_buffer import EpisodicReplayBuffer
from chainerrl import spaces
from chainerrl import v_functions


def phi(obs):
    return obs.astype(np.float32, copy=False)


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--t-max', type=int, default=50)
    parser.add_argument('--n-times-replay', type=int, default=4)
    parser.add_argument('--n-hidden-channels', type=int, default=100)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--replay-capacity', type=int, default=5000)
    parser.add_argument('--replay-start-size', type=int, default=10 ** 3)
    parser.add_argument('--disable-online-update', action='store_true')
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-2)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--truncation-threshold', type=float, default=5)
    parser.add_argument('--trust-region-delta', type=float, default=0.1)
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

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

    if isinstance(action_space, spaces.Box):
        model = acer.ACERSDNSeparateModel(
            pi=policies.FCGaussianPolicy(
                obs_space.low.size, action_space.low.size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers,
                bound_mean=True,
                min_action=action_space.low,
                max_action=action_space.high),
            v=v_functions.FCVFunction(
                obs_space.low.size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers),
            adv=q_functions.FCSAQFunction(
                obs_space.low.size, action_space.low.size,
                n_hidden_channels=args.n_hidden_channels // 4,
                n_hidden_layers=args.n_hidden_layers),
        )
    else:
        model = acer.ACERSeparateModel(
            pi=links.Sequence(
                L.Linear(obs_space.low.size, args.n_hidden_channels),
                F.relu,
                L.Linear(args.n_hidden_channels, action_space.n, wscale=1e-3),
                SoftmaxDistribution),
            q=links.Sequence(
                L.Linear(obs_space.low.size, args.n_hidden_channels),
                F.relu,
                L.Linear(args.n_hidden_channels, action_space.n, wscale=1e-3),
                DiscreteActionValue),
        )

    opt = rmsprop_async.RMSpropAsync(
        lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))

    replay_buffer = EpisodicReplayBuffer(args.replay_capacity)
    agent = acer.ACER(model, opt, t_max=args.t_max, gamma=0.99,
                      replay_buffer=replay_buffer,
                      n_times_replay=args.n_times_replay,
                      replay_start_size=args.replay_start_size,
                      disable_online_update=args.disable_online_update,
                      use_trust_region=True,
                      trust_region_delta=args.trust_region_delta,
                      truncation_threshold=args.truncation_threshold,
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
