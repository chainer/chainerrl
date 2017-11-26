"""An example of training DQN against OpenAI Gym Envs.

This script is an example of training a DQN agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported. For continuous action
spaces, A NAF (Normalized Advantage Function) is used to approximate Q-values.

To solve CartPole-v0, run:
    python train_dqn_gym.py --env CartPole-v0

To solve Pendulum-v0, run:
    python train_dqn_gym.py --env Pendulum-v0
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import argparse
import sys

from chainer import optimizers
import gym
gym.undo_logger_setup()
from gym import spaces
import gym.wrappers
import numpy as np

from chainerrl.agents.qr_dqn import QRDQN
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer


import chainer
import chainer.functions as F
from chainer import cuda
import chainerrl


class QuantileQFunction(chainer.Chain, chainerrl.q_function.StateQFunction):
    def __init__(
            self,
            obs_size, n_actions, n_diracs,
            n_hidden_channels,
            n_hidden_layers,
            ):
        self.n_actions = n_actions
        self.n_diracs = n_diracs
        super().__init__()
        with self.init_scope():
            self.model = chainerrl.links.MLP(
                obs_size, n_actions * n_diracs,
                hidden_sizes=[n_hidden_channels] * n_hidden_layers,
            )

    def __call__(self, x):
        y = self.model(x).reshape(-1, self.n_actions, self.n_diracs)
        return ActionQuantile(y)


# class ActionQuantile(chainerrl.action_value.ActionValue):
class ActionQuantile(chainerrl.action_value.DiscreteActionValue):

    def __init__(self, q_quantiles, q_values_formatter=lambda x: x):
        assert isinstance(q_quantiles, chainer.Variable)
        self.xp = cuda.get_array_module(q_quantiles.data)
        self.q_quantiles = q_quantiles
        self.n_actions = q_quantiles.data.shape[1]
        self.n_diracs = q_quantiles.data.shape[2]
        self.q_values_formatter = q_values_formatter

    @property
    def q_values(self):
        return F.mean(self.q_quantiles, axis=2)

    def evaluate_actions(self, actions):
        if isinstance(actions, chainer.Variable):
            actions = actions.data

        # print(self.xp.arange(actions.size))
        # print(actions)
        return self.q_quantiles[
            self.xp.arange(actions.size),
            actions
        ]

    # @property
    # def greedy_actions(self):



def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='dqn_out')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--steps', type=int, default=10 ** 5)
    parser.add_argument('--prioritized-replay', action='store_true')
    parser.add_argument('--episodic-replay', action='store_true')
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=10 ** 4)
    parser.add_argument('--n-hidden-channels', type=int, default=100)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--render-train', action='store_true')
    parser.add_argument('--render-eval', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    args = parser.parse_args()

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    def make_env(for_eval):
        env = gym.make(args.env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        if not for_eval:
            misc.env_modifiers.make_reward_filtered(
                env, lambda x: x * args.reward_scale_factor)
        if ((args.render_eval and for_eval) or
                (args.render_train and not for_eval)):
            misc.env_modifiers.make_rendered(env)
        return env

    env = make_env(for_eval=False)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_size = env.observation_space.low.size
    action_space = env.action_space

    if isinstance(action_space, spaces.Box):
        assert False
        action_size = action_space.low.size
        # Use NAF to apply DQN to continuous action spaces
        q_func = q_functions.FCQuadraticStateQFunction(
            obs_size, action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            action_space=action_space)
        # Use the Ornstein-Uhlenbeck process for exploration
        ou_sigma = (action_space.high - action_space.low) * 0.2
        explorer = explorers.AdditiveOU(sigma=ou_sigma)
    else:
        n_actions = action_space.n
        q_func = QuantileQFunction(
            obs_size, n_actions, n_diracs=8,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers)
        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
            action_space.sample)

    opt = optimizers.Adam()
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 5
    if args.episodic_replay:
        if args.minibatch_size is None:
            args.minibatch_size = 4
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) \
                // args.update_interval
            rbuf = replay_buffer.PrioritizedEpisodicReplayBuffer(
                rbuf_capacity, betasteps=betasteps)
        else:
            rbuf = replay_buffer.EpisodicReplayBuffer(rbuf_capacity)
    else:
        if args.minibatch_size is None:
            args.minibatch_size = 32
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) \
                // args.update_interval
            rbuf = replay_buffer.PrioritizedReplayBuffer(
                rbuf_capacity, betasteps=betasteps)
        else:
            rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

    def phi(obs):
        return obs.astype(np.float32)

    agent = QRDQN(q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
                explorer=explorer, replay_start_size=args.replay_start_size,
                target_update_interval=args.target_update_interval,
                update_interval=args.update_interval,
                phi=phi, minibatch_size=args.minibatch_size,
                target_update_method=args.target_update_method,
                soft_update_tau=args.soft_update_tau,
                episodic_update=args.episodic_replay, episodic_update_len=16)

    if args.load:
        agent.load(args.load)

    eval_env = make_env(for_eval=True)

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
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_env=eval_env,
            max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
