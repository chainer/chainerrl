from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from chainer import optimizers
import numpy as np

from chainerrl.agents.ddpg import DDPG
from chainerrl.agents.ddpg import DDPGModel
from chainerrl.envs.abc import ABC
from chainerrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from chainerrl.policy import FCBNDeterministicPolicy
from chainerrl.policy import FCLSTMDeterministicPolicy
from chainerrl.q_functions import FCBNLateActionSAQFunction
from chainerrl.q_functions import FCLSTMSAQFunction
from chainerrl import replay_buffer

from test_training import _TestTraining


class _TestDDPGOnABC(_TestTraining):

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
        return self.make_ddpg_agent(env=env, model=model,
                                    actor_opt=actor_opt, critic_opt=critic_opt,
                                    explorer=explorer, rbuf=rbuf, gpu=gpu)

    def make_ddpg_agent(self, env, model, actor_opt, critic_opt, explorer,
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


class _TestDDPGOnContinuousPOABC(_TestDDPGOnABC):

    def make_model(self, env):
        n_dim_obs = env.observation_space.low.size
        n_dim_action = env.action_space.low.size
        n_hidden_channels = 50
        policy = FCLSTMDeterministicPolicy(n_input_channels=n_dim_obs,
                                           n_hidden_layers=2,
                                           n_hidden_channels=n_hidden_channels,
                                           action_size=n_dim_action,
                                           min_action=env.action_space.low,
                                           max_action=env.action_space.high,
                                           bound_action=True)

        q_func = FCLSTMSAQFunction(n_dim_obs=n_dim_obs,
                                   n_dim_action=n_dim_action,
                                   n_hidden_layers=2,
                                   n_hidden_channels=n_hidden_channels)

        return DDPGModel(policy=policy, q_func=q_func)

    def make_env_and_successful_return(self, test):
        return ABC(discrete=False, partially_observable=True,
                   deterministic=test), 1

    def make_replay_buffer(self, env):
        return replay_buffer.EpisodicReplayBuffer(10 ** 5)


class _TestDDPGOnContinuousABC(_TestDDPGOnABC):

    def make_model(self, env):
        n_dim_obs = env.observation_space.low.size
        n_dim_action = env.action_space.low.size
        n_hidden_channels = 50
        policy = FCBNDeterministicPolicy(n_input_channels=n_dim_obs,
                                         n_hidden_layers=2,
                                         n_hidden_channels=n_hidden_channels,
                                         action_size=n_dim_action,
                                         min_action=env.action_space.low,
                                         max_action=env.action_space.high,
                                         bound_action=True)

        q_func = FCBNLateActionSAQFunction(n_dim_obs=n_dim_obs,
                                           n_dim_action=n_dim_action,
                                           n_hidden_layers=2,
                                           n_hidden_channels=n_hidden_channels)

        return DDPGModel(policy=policy, q_func=q_func)

    def make_env_and_successful_return(self, test):
        return ABC(discrete=False, deterministic=test), 1


class TestDDPGOnContinuousPOABC(_TestDDPGOnContinuousPOABC):

    def make_ddpg_agent(self, env, model, actor_opt, critic_opt, explorer,
                        rbuf, gpu):
        return DDPG(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
                    explorer=explorer, replay_start_size=100,
                    target_update_method='soft', target_update_frequency=1,
                    episodic_update=True, update_frequency=1)


class TestDDPGOnContinuousABC(_TestDDPGOnContinuousABC):

    def make_ddpg_agent(self, env, model, actor_opt, critic_opt, explorer,
                        rbuf, gpu):
        return DDPG(model, actor_opt, critic_opt, rbuf, gpu=gpu, gamma=0.9,
                    explorer=explorer, replay_start_size=100,
                    target_update_method='soft', target_update_frequency=1,
                    episodic_update=False)
