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
import numpy as np

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

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        if self.head:
            vs = next_target_action_value

            pi_values = self.model(batch_next_state)
            p_means = pi_values.q_values.data
            p_sigmas = pi_values.sigmas.data

            # normal distribution values + thompson sampling
            #starts = means.min(axis=1)-sigmas.max(axis=1)*3
            start = (p_means-p_sigmas*3).min(axis=1)
            #ends = means.max(axis=1)+sigmas.max(axis=1)*3#means.max()+sigmas.max()*3
            end = (p_means+p_sigmas*3).max(axis=1)

            def estimate(n):
                interval = (end-start)/n

                mean = 0
                sigma = 0
                act_probs = self.xp.ones((p_means.shape[0], p_means.shape[1]), dtype=self.xp.float32) * 1e-5

                for i in range(n):
                    alpha = start + interval*i

                    def get_prob(x):
                        pdfs = prob(x, p_means.flatten(), p_sigmas.flatten()).reshape((p_means.shape[0], p_means.shape[1]))
                        cdfs = cdf(x, p_means.flatten(), p_sigmas.flatten()).reshape((p_means.shape[0], p_means.shape[1]))

                        a_probs = self.xp.zeros_like(p_means)

                        for a in range(p_means.shape[1]):
                            p = pdfs[:, a]
                            for a2 in range(p_means.shape[1]):
                                if a2 != a:
                                    p *= cdfs[:, a2]

                            a_probs[:, a] = p.data

                        return a_probs

                    alpha = self.xp.repeat(alpha, 3)
                    est = get_prob(alpha)
                    act_probs += est

                act_probs /= act_probs.sum(axis=1)[:, None]

                #temp = 0.1
                #act_probs = self.xp.exp(act_probs/temp) / self.xp.sum(self.xp.exp(act_probs/temp), axis=1)[:, None]

                """
                counts = np.zeros((p_means.shape[0], 3))

                np_p_means = self.xp.asnumpy(p_means)
                np_p_sigmas = self.xp.asnumpy(p_sigmas)
                for b in range(p_means.shape[0]):
                    for i in range(1000):
                        samples = []

                        for a in range(3):
                            x = np.random.normal(np_p_means[b, a], np_p_sigmas[b, a])

                            samples.append(x)

                        win = np.argmax(np.asarray(samples))
                        counts[b, win] += 1
                """

                #print("dist", p_means[0], p_sigmas[0])
                #print("interval", start[0], end[0])
                #print("sampled", (counts / counts.sum(axis=1)[:, None])[0])
                #print("estimated", act_probs[0])

                #act_probs = self.xp.asarray(counts).astype(self.xp.float32)
                #act_probs /= act_probs.sum(axis=1)[:, None]

                mean = (vs.q_values.data * act_probs).sum(1)
                sigma = (vs.sigmas.data * act_probs).sum(1)

                #for i in range(n):
                #    diff = ((start + interval*i) - mean)**2.0
                #    sigma += diff*(int_p[i]*interval)

                return mean, sigma

            mean, sigma = estimate(10)

            #batch_next_action = exp_batch['next_action']
            #next_target_action_value = self.target_q_function(
            #    batch_next_state)
            #next_q = next_target_action_value.evaluate_actions(
            #    batch_next_action)
            #batch_rewards = exp_batch['reward']

            #mean = batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q
            #sigma = self.gamma * next_target_action_value.max_sigma

            mean = batch_rewards + self.gamma * (1.0 - batch_terminal) * mean
            sigma = self.gamma * sigma

            return mean[:, None], sigma[:, None]
        else:
            values = next_target_action_value.q_values
            # epsilon-greedy expectation
            max_prob = 1-self.explorer.epsilon
            pi_dist = self.xp.ones_like(values) * (self.explorer.epsilon/values.shape[1])
            pi_dist[self.xp.arange(pi_dist.shape[0]), greedy.data] += max_prob

            expected_q = F.sum(pi_dist * values, 1)
            return batch_rewards + self.gamma * (1.0 - batch_terminal) * expected_q
