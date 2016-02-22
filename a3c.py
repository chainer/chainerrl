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
            return action
        else:
            return None


def create_a3c_agents(num_agents):
    raise NotImplementedError


class A3CMaster(object):

    def __init__(self):
        raise NotImplementedError

    def create_agent(self):
        raise NotImplementedError
