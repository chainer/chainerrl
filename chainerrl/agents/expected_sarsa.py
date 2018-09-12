from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from chainerrl.agents import dqn
import chainer.functions as F

from chainer.functions.math import ndtr
from chainer.functions.math import exponential
import math

PROBC = 1. / (2 * math.pi) ** 0.5

def prob(x, loc, scale):
    return (PROBC / scale) * exponential.exp(
        - 0.5 * (x - loc) ** 2 / scale ** 2)

def cdf(x, loc, scale):
    return ndtr.ndtr((x - loc) / scale)

class ExpectedSARSA(dqn.DQN):
    """SARSA.

    Unlike DQN, this agent uses actions that have been actually taken to
    compute tareget Q values, thus is an on-policy algorithm.
    """

    def _compute_target_values(self, exp_batch, gamma):

        batch_next_state = exp_batch['next_state']

        next_target_action_value = self.target_q_function(
            batch_next_state)
        greedy = next_target_action_value.greedy_actions
        values = next_target_action_value.q_values

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        if self.head:
            means = values
            sigmas = next_target_action_value.sigmas

            # normal distribution values + thompson sampling
            start = values.data.min()
            end = values.data.max()

            def estimate(n):
                interval = (end-start)/n

                int_p = []
                mean = 0
                sigma = 0

                for i in range(n):
                    alpha = start + interval*i

                    def get_prob(x):
                        pdfs = prob(x, means.data.flatten(), sigmas.data.flatten()).reshape((means.shape[0], means.shape[1]))
                        cdfs = cdf(x, means.data.flatten(), sigmas.data.flatten()).reshape((means.shape[0], means.shape[1]))

                        probs = 0

                        for a in range(values.shape[1]):
                            p = pdfs[:, a]
                            for a2 in range(values.shape[1]):
                                if a2 != a:
                                    p *= cdfs[:, a]
                            probs += p

                        return probs

                    p = get_prob(alpha)
                    int_p.append(p)
                    mean += alpha*(p*interval)

                for i in range(n):
                    diff = ((start + interval*i) - mean)**2.0
                    sigma += diff*(int_p[i]*interval)

                return mean, sigma

            mean, sigma = estimate(10)

            #mean = batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q
            #sigma = self.gamma * next_target_action_value.max_sigma
            return mean[:, None], self.gamma * sigma[:, None]
        else:
            # epsilon-greedy expectation
            max_prob = 1-self.explorer.epsilon
            pi_dist = self.xp.ones_like(values) * (self.explorer.epsilon/values.shape[1])
            pi_dist[self.xp.arange(pi_dist.shape[0]), greedy.data] += max_prob

            expected_q = F.sum(pi_dist * values, 1)
            return batch_rewards + self.gamma * (1.0 - batch_terminal) * expected_q
