"""An example of training PCL against OpenAI Gym Envs.

This script is an example of training a PCL agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_pcl_gym.py

To solve InvertedPendulum-v1, run:
    python train_pcl_gym.py --env InvertedPendulum-v1
"""
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
import gym
import gym.spaces
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl import experiments
from chainerrl import misc
from chainerrl.optimizers import rmsprop_async


def exp_return_of_episode(episode):
    return np.exp(sum(x['reward'] for x in episode))


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--rollout-len', type=int, default=10)
    parser.add_argument('--n-hidden-channels', type=int, default=100)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--n-times-replay', type=int, default=1)
    parser.add_argument('--replay-start-size', type=int, default=10000)
    parser.add_argument('--t-max', type=int, default=None)
    parser.add_argument('--tau', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--train-async', action='store_true', default=False)
    parser.add_argument('--prioritized-replay', action='store_true',
                        default=False)
    parser.add_argument('--disable-online-update', action='store_true',
                        default=False)
    parser.add_argument('--backprop-future-values', action='store_true',
                        default=True)
    parser.add_argument('--no-backprop-future-values', action='store_false',
                        dest='backprop_future_values')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    # Set a random seed used in ChainerRL.
    # If you use async training (--train-async), the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    if args.train_async:
        # Set different random seeds for different subprocesses.
        # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
        # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
        process_seeds = np.arange(args.processes) + args.seed * args.processes
        assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        if args.train_async:
            process_seed = int(process_seeds[process_idx])
            env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        else:
            env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
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

    # Switch policy types accordingly to action space types
    if isinstance(action_space, gym.spaces.Box):
        model = chainerrl.agents.pcl.PCLSeparateModel(
            pi=chainerrl.policies.FCGaussianPolicy(
                obs_space.low.size, action_space.low.size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers,
                bound_mean=True,
                min_action=action_space.low,
                max_action=action_space.high,
                var_wscale=1e-3,
                var_bias=1,
                var_type='diagonal',
            ),
            v=chainerrl.v_functions.FCVFunction(
                obs_space.low.size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers,
            )
        )
    else:
        model = chainerrl.agents.pcl.PCLSeparateModel(
            pi=chainerrl.policies.FCSoftmaxPolicy(
                obs_space.low.size, action_space.n,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers
            ),
            v=chainerrl.v_functions.FCVFunction(
                obs_space.low.size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=args.n_hidden_layers,
            ),
        )

    if not args.train_async and args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    if args.train_async:
        opt = rmsprop_async.RMSpropAsync(lr=args.lr, alpha=0.99)
    else:
        opt = chainer.optimizers.Adam(alpha=args.lr)
    opt.setup(model)

    if args.prioritized_replay:
        replay_buffer = \
            chainerrl.replay_buffer.PrioritizedEpisodicReplayBuffer(
                capacity=5 * 10 ** 3,
                uniform_ratio=0.1,
                default_priority_func=exp_return_of_episode,
                wait_priority_after_sampling=False,
                return_sample_weights=False)
    else:
        replay_buffer = chainerrl.replay_buffer.EpisodicReplayBuffer(
            capacity=5 * 10 ** 3)

    agent = chainerrl.agents.PCL(
        model, opt, replay_buffer=replay_buffer,
        t_max=args.t_max, gamma=0.99,
        tau=args.tau,
        rollout_len=args.rollout_len,
        n_times_replay=args.n_times_replay,
        replay_start_size=args.replay_start_size,
        batchsize=args.batchsize,
        train_async=args.train_async,
        disable_online_update=args.disable_online_update,
        backprop_future_values=args.backprop_future_values,
    )
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
        if args.train_async:
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
        else:
            experiments.train_agent_with_evaluation(
                agent=agent,
                env=make_env(0, test=False),
                eval_env=make_env(0, test=True),
                outdir=args.outdir,
                steps=args.steps,
                eval_n_steps=None,
                eval_n_episodes=args.eval_n_runs,
                eval_interval=args.eval_interval,
                train_max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
