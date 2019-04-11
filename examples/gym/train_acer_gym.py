from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

import chainer
from chainer import functions as F
from chainer.initializers import LeCunNormal
from chainer import links as L
import gym
from gym import spaces
import gym.wrappers
import numpy as np

import chainerrl
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
from chainerrl import v_functions


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
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
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
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

    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor and process_idx == 0:
            env = gym.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render and process_idx == 0 and not test:
            env = chainerrl.wrappers.Render(env)
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
                L.Linear(args.n_hidden_channels, action_space.n,
                         initialW=LeCunNormal(1e-3)),
                SoftmaxDistribution),
            q=links.Sequence(
                L.Linear(obs_space.low.size, args.n_hidden_channels),
                F.relu,
                L.Linear(args.n_hidden_channels, action_space.n,
                         initialW=LeCunNormal(1e-3)),
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
                      beta=args.beta)
    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
