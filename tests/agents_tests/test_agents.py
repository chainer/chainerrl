from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

from chainer import optimizers
from chainer import testing
import gym
import gym.spaces

import basetest_agents as base
from chainerrl import agents
from chainerrl import explorers
from chainerrl import policies
from chainerrl import q_functions
from chainerrl import replay_buffer
from chainerrl import v_function


def create_stochastic_policy_for_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    ndim_obs = env.observation_space.low.size
    if isinstance(env.action_space, gym.spaces.Discrete):
        return policies.FCSoftmaxPolicy(ndim_obs, env.action_space.n)
    elif isinstance(env.action_space, gym.spaces.Box):
        return policies.FCGaussianPolicy(
            ndim_obs, env.action_space.low.size,
            bound_mean=False)
    else:
        raise NotImplementedError()


def create_deterministic_policy_for_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    ndim_obs = env.observation_space.low.size
    return policies.FCDeterministicPolicy(
        n_input_channels=ndim_obs,
        action_size=env.action_space.low.size,
        n_hidden_channels=200,
        n_hidden_layers=2,
        bound_action=False)


def create_state_q_function_for_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    ndim_obs = env.observation_space.low.size
    if isinstance(env.action_space, gym.spaces.Discrete):
        return q_functions.FCStateQFunctionWithDiscreteAction(
            ndim_obs=ndim_obs,
            n_actions=env.action_space.n,
            n_hidden_channels=200,
            n_hidden_layers=2)
    elif isinstance(env.action_space, gym.spaces.Box):
        return q_functions.FCQuadraticStateQFunction(
            n_input_channels=ndim_obs,
            n_dim_action=env.action_space.low.size,
            n_hidden_channels=200,
            n_hidden_layers=2,
            action_space=env.action_space)
    else:
        raise NotImplementedError()


def create_state_action_q_function_for_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    ndim_obs = env.observation_space.low.size
    return q_functions.FCSAQFunction(
        n_dim_obs=ndim_obs,
        n_dim_action=env.action_space.low.size,
        n_hidden_channels=200,
        n_hidden_layers=2)


def create_v_function_for_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    ndim_obs = env.observation_space.low.size
    return v_function.FCVFunction(ndim_obs)


@testing.parameterize(*testing.product({
    'discrete': [True, False],
    'partially_observable': [False],
    'episodic': [False],
}))
class TestA3C(base._TestAgentInterface):

    def create_agent(self, env):
        model = agents.a3c.A3CSeparateModel(
            pi=create_stochastic_policy_for_env(env),
            v=create_v_function_for_env(env))
        opt = optimizers.Adam()
        opt.setup(model)
        return agents.A3C(model, opt, t_max=1, gamma=0.99)


@testing.parameterize(*testing.product({
    'discrete': [True],
    'partially_observable': [False],
    'episodic': [False],
}))
class TestACER(base._TestAgentInterface):

    def create_agent(self, env):
        model = agents.acer.ACERSeparateModel(
            pi=create_stochastic_policy_for_env(env),
            q=create_state_q_function_for_env(env))
        opt = optimizers.Adam()
        opt.setup(model)
        rbuf = replay_buffer.EpisodicReplayBuffer(10 ** 4)
        return agents.ACER(model, opt, t_max=1, gamma=0.99,
                           replay_buffer=rbuf)


@testing.parameterize(*testing.product({
    'discrete': [True, False],
    'partially_observable': [False],
    'episodic': [False],
}))
class TestDQN(base._TestAgentInterface):

    def create_agent(self, env):
        model = create_state_q_function_for_env(env)
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        opt = optimizers.Adam()
        opt.setup(model)
        explorer = explorers.ConstantEpsilonGreedy(
            0.2, random_action_func=lambda: env.action_space.sample())
        return agents.DQN(model, opt, rbuf, gamma=0.99, explorer=explorer)


@testing.parameterize(*testing.product({
    'discrete': [True, False],
    'partially_observable': [False],
    'episodic': [False],
}))
class TestDoubleDQN(base._TestAgentInterface):

    def create_agent(self, env):
        model = create_state_q_function_for_env(env)
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        opt = optimizers.Adam()
        opt.setup(model)
        explorer = explorers.ConstantEpsilonGreedy(
            0.2, random_action_func=lambda: env.action_space.sample())
        return agents.DoubleDQN(
            model, opt, rbuf, gamma=0.99, explorer=explorer)


@testing.parameterize(*testing.product({
    'discrete': [True, False],
    'partially_observable': [False],
    'episodic': [False],
}))
class TestNSQ(base._TestAgentInterface):

    def create_agent(self, env):
        model = create_state_q_function_for_env(env)
        opt = optimizers.Adam()
        opt.setup(model)
        explorer = explorers.ConstantEpsilonGreedy(
            0.2, random_action_func=lambda: env.action_space.sample())
        return agents.NSQ(
            q_function=model,
            optimizer=opt,
            t_max=1,
            gamma=0.99,
            i_target=100,
            explorer=explorer)


@testing.parameterize(*testing.product({
    'discrete': [False],
    'partially_observable': [False],
    'episodic': [False],
}))
class TestDDPG(base._TestAgentInterface):

    def create_agent(self, env):
        model = agents.ddpg.DDPGModel(
            policy=create_deterministic_policy_for_env(env),
            q_func=create_state_action_q_function_for_env(env))
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        opt_a = optimizers.Adam()
        opt_a.setup(model.policy)
        opt_b = optimizers.Adam()
        opt_b.setup(model.q_function)
        explorer = explorers.AdditiveGaussian(scale=1)
        return agents.DDPG(model, opt_a, opt_b, rbuf, gamma=0.99,
                           explorer=explorer)


@testing.parameterize(*testing.product({
    'discrete': [False],
    'partially_observable': [False],
    'episodic': [False],
}))
class TestPGT(base._TestAgentInterface):

    def create_agent(self, env):
        model = agents.ddpg.DDPGModel(
            policy=create_stochastic_policy_for_env(env),
            q_func=create_state_action_q_function_for_env(env))
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        opt_a = optimizers.Adam()
        opt_a.setup(model.policy)
        opt_b = optimizers.Adam()
        opt_b.setup(model.q_function)
        explorer = explorers.AdditiveGaussian(scale=1)
        return agents.PGT(model, opt_a, opt_b, rbuf, gamma=0.99,
                          explorer=explorer)
