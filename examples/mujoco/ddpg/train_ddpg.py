"""A training script of DDPG on OpenAI Gym Mujoco environments.

This script follows the settings of http://arxiv.org/abs/1802.09477 as much
as possible.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import logging
import os
import sys

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import gym
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents.ddpg import DDPG
from chainerrl.agents.ddpg import DDPGModel
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer


def concat_obs_and_action(obs, action):
    """Concat observation and action to feed the critic."""
    return F.concat((obs, action), axis=-1)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='Hopper-v2',
                        help='OpenAI Gym MuJoCo env to perform algorithm on.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--load', type=str, default='',
                        help='Directory to load agent from.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--eval-n-runs', type=int, default=10,
                        help='Number of episodes run for each evaluation.')
    parser.add_argument('--eval-interval', type=int, default=5000,
                        help='Interval in timesteps between evaluations.')
    parser.add_argument('--replay-start-size', type=int, default=10000,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Minibatch size')
    parser.add_argument('--render', action='store_true',
                        help='Render env states in a GUI window.')
    parser.add_argument('--demo', action='store_true',
                        help='Just run evaluation, not training.')
    parser.add_argument('--monitor', action='store_true',
                        help='Wrap env with gym.wrappers.Monitor.')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    def make_env(test):
        env = gym.make(args.env)
        # Unwrap TimiLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if args.render and not test:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = env.observation_space
    action_space = env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)

    action_size = action_space.low.size

    winit = chainer.initializers.LeCunUniform(3 ** -0.5)
    q_func = chainer.Sequential(
        concat_obs_and_action,
        L.Linear(None, 400, initialW=winit),
        F.relu,
        L.Linear(None, 300, initialW=winit),
        F.relu,
        L.Linear(None, 1, initialW=winit),
    )
    policy = chainer.Sequential(
        L.Linear(None, 400, initialW=winit),
        F.relu,
        L.Linear(None, 300, initialW=winit),
        F.relu,
        L.Linear(None, action_size, initialW=winit),
        F.tanh,
        chainerrl.distribution.ContinuousDeterministicDistribution,
    )
    model = DDPGModel(q_func=q_func, policy=policy)

    # Draw the computational graph and save it in the output directory.
    fake_obs = chainer.Variable(
        model.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
        name='observation')
    fake_action = chainer.Variable(
        model.xp.zeros_like(action_space.low, dtype=np.float32)[None],
        name='action')
    chainerrl.misc.draw_computational_graph(
        [policy(fake_obs)], os.path.join(args.outdir, 'policy'))
    chainerrl.misc.draw_computational_graph(
        [q_func(fake_obs, fake_action)], os.path.join(args.outdir, 'q_func'))

    opt_a = optimizers.Adam()
    opt_c = optimizers.Adam()
    opt_a.setup(model['policy'])
    opt_c.setup(model['q_function'])

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = explorers.AdditiveGaussian(
        scale=0.1, low=action_space.low, high=action_space.high)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(
            action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = DDPG(
        model,
        opt_a,
        opt_c,
        rbuf,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_method='soft',
        target_update_interval=1,
        update_interval=1,
        soft_update_tau=5e-3,
        n_times_update=1,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
    )

    if len(args.load) > 0:
        agent.load(args.load)

    eval_env = make_env(test=True)
    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_env=eval_env, eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir,
            train_max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
