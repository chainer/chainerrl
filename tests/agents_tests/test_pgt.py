from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np

from chainerrl.agents.pgt import PGT
from chainerrl.envs.abc import ABC
from chainerrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from chainerrl.links import Sequence
from chainerrl import policies
from chainerrl import q_function
from chainerrl import replay_buffer

from test_training import _TestTraining


class _TestPGTOnABC(_TestTraining):

    def make_agent(self, env, gpu):
        model = self.make_model(env)
        policy = model['policy']
        q_func = model['q_function']

        actor_opt = optimizers.Adam(alpha=1e-4)
        actor_opt.setup(policy)

        critic_opt = optimizers.Adam(alpha=1e-3)
        critic_opt.setup(q_func)

        explorer = self.make_explorer(env)
        rbuf = self.make_replay_buffer(env)
        return self.make_pgt_agent(env=env, model=model,
                                   actor_opt=actor_opt, critic_opt=critic_opt,
                                   explorer=explorer, rbuf=rbuf, gpu=gpu)

    def make_pgt_agent(self, env, model, actor_opt, critic_opt, explorer,
                       rbuf, gpu):
        raise NotImplementedError()

    def make_explorer(self, env):
        def random_action_func():
            a = env.action_space.sample()
            if isinstance(a, np.ndarray):
                return a.astype(np.float32)
            else:
                return a
        return LinearDecayEpsilonGreedy(1.0, 0.2, 1000, random_action_func)

    def make_replay_buffer(self, env):
        return replay_buffer.ReplayBuffer(10 ** 5)


class _TestPGTOnContinuousPOABC(_TestPGTOnABC):

    def make_model(self, env):
        n_dim_obs = env.observation_space.low.size
        n_dim_action = env.action_space.low.size
        n_hidden_channels = 50
        policy = Sequence(
            L.Linear(n_dim_obs, n_hidden_channels),
            F.relu,
            L.Linear(n_hidden_channels, n_hidden_channels),
            F.relu,
            L.LSTM(n_hidden_channels, n_hidden_channels),
            policies.FCGaussianPolicy(
                n_input_channels=n_hidden_channels,
                action_size=n_dim_action,
                min_action=env.action_space.low,
                max_action=env.action_space.high)
        )

        q_func = q_function.FCLSTMSAQFunction(
            n_dim_obs=n_dim_obs,
            n_dim_action=n_dim_action,
            n_hidden_layers=2,
            n_hidden_channels=n_hidden_channels)

        return chainer.Chain(policy=policy, q_function=q_func)

    def make_env_and_successful_return(self, test):
        return ABC(discrete=False, partially_observable=True,
                   deterministic=test), 1

    def make_replay_buffer(self, env):
        return replay_buffer.EpisodicReplayBuffer(10 ** 5)


class _TestPGTOnContinuousABC(_TestPGTOnABC):

    def make_model(self, env):
        n_dim_obs = env.observation_space.low.size
        n_dim_action = env.action_space.low.size
        n_hidden_channels = 50

        policy = policies.FCGaussianPolicy(
            n_input_channels=n_dim_obs,
            n_hidden_layers=2,
            n_hidden_channels=n_hidden_channels,
            action_size=n_dim_action,
            min_action=env.action_space.low,
            max_action=env.action_space.high)

        q_func = q_function.FCSAQFunction(
            n_dim_obs=n_dim_obs,
            n_dim_action=n_dim_action,
            n_hidden_layers=2,
            n_hidden_channels=n_hidden_channels)

        return chainer.Chain(policy=policy, q_function=q_func)

    def make_env_and_successful_return(self, test):
        return ABC(discrete=False, deterministic=test), 1


# Currently PGT does not support recurrent models
# class TestPGTOnContinuousPOABC(_TestPGTOnContinuousPOABC):
#
#     def make_pgt_agent(self, env, model, actor_opt, critic_opt, explorer,
#                        rbuf, gpu):
#         return PGT(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
#                    explorer=explorer, replay_start_size=100,
#                    target_update_method='soft', target_update_frequency=1,
#                    episodic_update=True, update_frequency=1,
#                    act_deterministically=True)


class TestPGTOnContinuousABC(_TestPGTOnContinuousABC):

    def make_pgt_agent(self, env, model, actor_opt, critic_opt, explorer,
                       rbuf, gpu):
        return PGT(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
                   explorer=explorer, replay_start_size=100,
                   target_update_method='soft', target_update_frequency=1,
                   act_deterministically=True)
