"""An example of training TRPO against OpenAI Gym Envs.

This script is an example of training a TRPO agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.

Chainer v3.0.0 or newer is required.

To solve CartPole-v0, run:
    python train_trpo_gym.py --env CartPole-v0 --steps 100000

To solve InvertedPendulum-v1, run:
    python train_trpo_gym.py --env InvertedPendulum-v1 --steps 100000
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import argparse
import logging
import os

import chainer
from chainer import functions as F
import gym
gym.undo_logger_setup()
import gym.wrappers
import numpy as np

import chainerrl


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID. Set to -1 to use CPUs only.')
    parser.add_argument('--env', type=str, default='Hopper-v1',
                        help='Gym Env ID')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total time steps for training.')
    parser.add_argument('--eval-interval', type=int, default=10000,
                        help='Interval between evaluation phases in steps.')
    parser.add_argument('--eval-n-runs', type=int, default=10,
                        help='Number of episodes ran in an evaluation phase')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the env')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Run demo episodes, not training')
    parser.add_argument('--load', type=str, default='',
                        help='Directory path to load a saved agent data from'
                             ' if it is a non-empty string.')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor the env by gym.wrappers.Monitor.'
                             ' Videos and additional log will be saved.')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    # Set random seed
    chainerrl.misc.set_random_seed(args.seed)

    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir)

    def make_env(test):
        env = gym.make(args.env)
        if args.seed is not None:
            env.seed(args.seed)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if args.render:
            chainerrl.misc.env_modifiers.make_rendered(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = env.observation_space
    action_space = env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)

    if not isinstance(obs_space, gym.spaces.Box):
        print("""\
This example only supports gym.spaces.Box observation spaces. To apply it to
other observation spaces, use a custom phi function that convert an observation
to numpy.ndarray of numpy.float32.""")  # NOQA
        return

    if isinstance(action_space, gym.spaces.Box):
        # Use a Gaussian policy for continuous action spaces
        policy = \
            chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance(
                obs_space.low.size,
                action_space.low.size,
                n_hidden_channels=64,
                n_hidden_layers=2,
                mean_wscale=0.01,
                nonlinearity=F.tanh,
                var_type='diagonal',
            )
    elif isinstance(action_space, gym.spaces.Discrete):
        # Use a Softmax policy for discrete action spaces
        policy = chainerrl.policies.FCSoftmaxPolicy(
            obs_space.low.size,
            action_space.n,
            n_hidden_channels=64,
            n_hidden_layers=2,
            last_wscale=0.01,
            nonlinearity=F.tanh,
        )
    else:
        print("""\
TRPO only supports gym.spaces.Box or gym.spaces.Discrete action spaces.""")  # NOQA
        return

    # Use a value function to reduce variance
    vf = chainerrl.v_functions.FCVFunction(
        obs_space.low.size,
        n_hidden_channels=64,
        n_hidden_layers=2,
        last_wscale=0.01,
        nonlinearity=F.tanh,
    )

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        policy.to_gpu(args.gpu)
        vf.to_gpu(args.gpu)

    # TRPO's policy is optimized via CG and line search, so it doesn't require
    # a chainer.Optimizer. Only the value function needs it.
    vf_opt = chainer.optimizers.Adam()
    vf_opt.setup(vf)

    # Draw the computational graph and save it in the output directory.
    fake_obs = chainer.Variable(
        policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
        name='observation')
    chainerrl.misc.draw_computational_graph(
        [policy(fake_obs)], os.path.join(args.outdir, 'policy'))
    chainerrl.misc.draw_computational_graph(
        [vf(fake_obs)], os.path.join(args.outdir, 'vf'))

    agent = chainerrl.agents.TRPO(
        policy=policy,
        vf=vf,
        vf_optimizer=vf_opt,
        phi=lambda x: x.astype(np.float32, copy=False),
        update_interval=1024,
        conjugate_gradient_damping=1e-1,
        gamma=0.99,
        lambd=0.98,
        vf_epochs=5,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(test=True)
        eval_stats = chainerrl.experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:

        chainerrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=make_env(test=True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit,
        )


if __name__ == '__main__':
    main()
