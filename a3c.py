from chainer import functions as F

import agent


class A3C(agent.Agent):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self, policy, v_function, optimizer, t_max, gamma):
        self.policy = policy
        self.v_function = v_function
        self.optimizer = optimizer
        self.t_max = t_max
        self.gamma = gamma
        self.t = 0
        self.t_start = 0
        self.past_action_prob = {}
        self.past_states = {}
        self.past_rewards = {}

    def act(self, state, reward, is_state_terminal):

        state = state.reshape((1,) + state.shape)

        self.past_rewards[self.t - 1] = reward

        if is_state_terminal or self.t - self.t_start == self.t_max:
            # Update
            if is_state_terminal:
                R = 0
            else:
                R = float(self.v_function(state).data)

            self.optimizer.zero_grads()

            for i in reversed(xrange(self.t_start, self.t)):
                R *= self.gamma
                R += self.past_rewards[i]
                v = self.v_function(self.past_states[i])
                advantage = R - v
                # Accumulate gradients of policy
                log_prob = F.log(self.past_action_prob[i])
                (- log_prob * float(advantage.data)).backward()
                # Accumulate gradients of value function
                (advantage ** 2).backward()

            self.optimizer.update()

            self.past_action_prob = {}
            self.past_states = {}
            self.past_rewards = {}

            self.t_start = self.t

        if not is_state_terminal:
            self.past_states[self.t] = state
            action, prob = self.policy.sample_with_probability(state)
            self.past_action_prob[self.t] = prob
            self.t += 1
            return action[0]
        else:
            return None

    @property
    def links(self):
        return [self.policy, self.v_function]

    @property
    def optimizers(self):
        return [self.optimizer]


def set_shared_params(a, b):
    for param_name, param in a.namedparams():
        if param_name in b:
            shared_param = b[param_name]
            param.data = np.frombuffer(shared_param.get_obj(
            ), dtype=param.data.dtype).reshape(param.data.shape)


def set_shared_states(a, b):
    for state_name, shared_state in b.iteritems():
        for param_name, param in shared_state.iteritems():
            old_param = a._states[state_name][param_name]
            a._states[state_name][param_name] = np.frombuffer(
                param.get_obj(),
                dtype=old_param.dtype).reshape(old_param.shape)


def extract_params_as_shared_arrays(link):
    shared_arrays = {}
    for param_name, param in link.namedparams():
        shared_arrays[param_name] = mp.Array('f', param.data.ravel())
    return shared_arrays


def extract_states_as_shared_arrays(optimizer):
    shared_arrays = {}
    for state_name, state in optimizer._states.iteritems():
        shared_arrays[state_name] = {}
        for param_name, param in state.iteritems():
            shared_arrays[state_name][
                param_name] = mp.Array('f', param.ravel())
    return shared_arrays


def create_a3c_agents(base_agent, num_process, envs):
    raise NotImplementedError


class A3CMaster(object):

    def __init__(self):
        raise NotImplementedError

    def create_agent(self):
        raise NotImplementedError
