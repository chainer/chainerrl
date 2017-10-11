"""An example of training A2C against OpenAI Gym Envs.

This script is an example of training a A2C agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_a2c_gym.py 8 --env CartPole-v0
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
from chainer import functions as F
import gym
import gym.wrappers
import numpy as np

from chainerrl.agents import a2c
from chainerrl.envs.vec_env import VectorEnv
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import policies
from chainerrl import v_function


def phi(obs):
    return obs.astype(np.float32)


class A2CFFSoftmax(chainer.ChainList, a2c.A2CModel):
    """An example of A2C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(64, 64)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A2CFFMellowmax(chainer.ChainList, a2c.A2CModel):
    """An example of A2C feedforward mellowmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(64, 64)):
        self.pi = policies.MellowmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A2CGaussian(chainer.ChainList, a2c.A2CModel):
    """An example of A2C recurrent Gaussian policy."""

    def __init__(self, obs_size, action_size):
        self.pi = policies.FCGaussianPolicyWithFixedCovariance(
            obs_size,
            action_size,
            np.log(np.e - 1),
            n_hidden_layers=2,
            n_hidden_channels=64,
            nonlinearity=F.tanh)
        self.v = v_function.FCVFunction(obs_size, n_hidden_layers=2,
                                        n_hidden_channels=64,
                                        nonlinearity=F.tanh)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


def main():

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--arch', type=str, default='Gaussian',
                        choices=('FFSoftmax', 'FFMellowmax', 'Gaussian'))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--update-steps', type=int, default=5)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-5)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer alpha')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
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

    sample_env = VectorEnv([
        make_env(i, args.demo) for i in range(args.processes)])
    if args.seed is not None:
        sample_env.seed(args.seed)

    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    # Switch policy types accordingly to action space types
    if args.arch == 'Gaussian':
        model = A2CGaussian(obs_space.low.size, action_space.low.size)
    elif args.arch == 'FFSoftmax':
        model = A2CFFSoftmax(obs_space.low.size, action_space.n)
    elif args.arch == 'FFMellowmax':
        model = A2CFFMellowmax(obs_space.low.size, action_space.n)

    optimizer = chainer.optimizers.RMSprop(args.lr,
                                           eps=args.rmsprop_epsilon,
                                           alpha=args.alpha)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.max_grad_norm))
    if args.weight_decay > 0:
        optimizer.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = a2c.A2C(model, optimizer, gamma=args.gamma,
                    gpu=args.gpu,
                    num_processes=args.processes,
                    update_steps=args.update_steps,
                    phi=phi,
                    use_gae=args.use_gae,
                    tau=args.tau)
    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        eval_env = make_env(0, True)
        experiments.train_agent_with_evaluation_sync(
            agent=agent, env=sample_env, steps=args.steps,
            log_interval=args.log_interval,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_env=eval_env)


if __name__ == '__main__':
    main()
