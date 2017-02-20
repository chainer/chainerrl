from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from chainer import optimizers
from chainer import testing
import numpy as np

from chainerrl.envs.abc import ABC
from chainerrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from chainerrl import q_functions
from chainerrl import replay_buffer

from test_training import _TestTraining


class _TestDQNLike(_TestTraining):

    def make_agent(self, env, gpu):
        q_func = self.make_q_func(env)
        opt = self.make_optimizer(env, q_func)
        explorer = self.make_explorer(env)
        rbuf = self.make_replay_buffer(env)
        agent = self.make_dqn_agent(env=env, q_func=q_func, opt=opt,
                                    explorer=explorer, rbuf=rbuf, gpu=gpu)
        return agent

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        raise NotImplementedError()

    def make_env_and_successful_return(self, test):
        raise NotImplementedError()

    def make_explorer(self, env):
        raise NotImplementedError()

    def make_optimizer(self, env, q_func):
        raise NotImplementedError()

    def make_replay_buffer(self, env):
        raise NotImplementedError()

    @testing.attr.slow
    @testing.attr.gpu
    def test_training_gpu(self):
        self._test_training(0, steps=1000)
        self._test_training(0, steps=0, load_model=True)

    @testing.attr.slow
    def test_training_cpu(self):
        self._test_training(-1, steps=1000)
        self._test_training(-1, steps=0, load_model=True)


class _TestDQNOnABC(_TestDQNLike):

    def make_agent(self, env, gpu):
        q_func = self.make_q_func(env)
        opt = self.make_optimizer(env, q_func)
        explorer = self.make_explorer(env)
        rbuf = self.make_replay_buffer(env)
        return self.make_dqn_agent(env=env, q_func=q_func,
                                   opt=opt, explorer=explorer, rbuf=rbuf,
                                   gpu=gpu)

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        raise NotImplementedError()

    def make_explorer(self, env):
        def random_action_func():
            a = env.action_space.sample()
            if isinstance(a, np.ndarray):
                return a.astype(np.float32)
            else:
                return a
        return LinearDecayEpsilonGreedy(1.0, 0.1, 1000, random_action_func)

    def make_optimizer(self, env, q_func):
        opt = optimizers.Adam()
        opt.setup(q_func)
        return opt

    def make_replay_buffer(self, env):
        return replay_buffer.ReplayBuffer(10 ** 5)


class _TestDQNOnDiscreteABC(_TestDQNOnABC):

    def make_q_func(self, env):
        return q_functions.FCStateQFunctionWithDiscreteAction(
            env.observation_space.low.size, env.action_space.n, 10, 10)

    def make_env_and_successful_return(self, test):
        return ABC(discrete=True, deterministic=test), 1


class _TestDQNOnDiscretePOABC(_TestDQNOnABC):

    def make_q_func(self, env):
        return q_functions.FCLSTMStateQFunction(
            n_dim_obs=env.observation_space.low.size,
            n_dim_action=env.action_space.n,
            n_hidden_channels=10,
            n_hidden_layers=1)

    def make_replay_buffer(self, env):
        return replay_buffer.EpisodicReplayBuffer(10 ** 5)

    def make_env_and_successful_return(self, test):
        return ABC(discrete=True, partially_observable=True,
                   deterministic=test), 1


class _TestDQNOnContinuousABC(_TestDQNOnABC):

    def make_q_func(self, env):
        n_dim_action = env.action_space.low.size
        n_dim_obs = env.observation_space.low.size
        return q_functions.FCQuadraticStateQFunction(
            n_input_channels=n_dim_obs,
            n_dim_action=n_dim_action,
            n_hidden_channels=20,
            n_hidden_layers=2,
            action_space=env.action_space)

    def make_env_and_successful_return(self, test):
        return ABC(discrete=False, deterministic=test), 1
