from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from chainerrl.agents import dqn
import chainer.functions as F
from chainerrl.agents.util import estimate

from chainer.functions.math import ndtr
from chainer.functions.math import exponential
import math
import numpy as np
import cv2

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
pltcanvas = FigureCanvas(fig)

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

            def sample(num_samples):
                counts = np.zeros((p_means.shape[0], 3))

                try:
                    np_p_means = self.xp.asnumpy(p_means)
                    np_p_sigmas = self.xp.asnumpy(p_sigmas)
                except:
                    np_p_means = p_means
                    np_p_sigmas = p_sigmas

                for b in range(p_means.shape[0]):
                    for i in range(num_samples):
                        samples = []

                        for a in range(3):
                            x = np.random.normal(np_p_means[b, a], np_p_sigmas[b, a])

                            samples.append(x)

                        win = np.argmax(np.asarray(samples))
                        counts[b, win] += 1

                #print(counts)
                act_probs = self.xp.asarray(counts).astype(self.xp.float32)
                act_probs /= act_probs.sum(axis=1)[:, None]

                return act_probs

            act_probs = estimate(self.xp, p_means, p_sigmas, 10, 3)
            #act_probs2 = estimate(10, 5)
            act_probs3 = sample(self.samples)

            #print("dist", p_means[0], p_sigmas[0])
            #print("interval", start[0], end[0])
            #print("sampled", act_probs[0])
            #print("estimated", act_probs2[0])

            """
            try:
                self.est_error = self.est_error * 0.99 + (1-0.99) * self.xp.asnumpy(((act_probs-act_probs3)**2).mean())
            except:
                self.est_error = self.est_error * 0.99 + (1-0.99) * ((act_probs-act_probs3)**2).mean()

            import matplotlib.pyplot as plt
            gca = fig.gca()
            edges = [0, 1, 1, 2, 2, 3]

            #ax1.autoscale(axis='y')
            ax1.set_title("p(Q|s) step: " + str(self.t))
            ax1.scatter(range(3), p_means[0])
            for i in range(3):
                ax1.plot([i, i], [p_means[0][i]-p_sigmas[0][i], p_means[0][i]+p_sigmas[0][i]])

            ax2.set_title("p(a|s)")
            ax2.plot(edges, act_probs[0][[0, 0, 1, 1, 2, 2]], label="estimate 10")
            #gca.plot(edges, act_probs2[0][[0, 0, 1, 1, 2, 2]], label="estimate 5")
            ax2.plot(edges, act_probs3[0][[0, 0, 1, 1, 2, 2]], label="sample 100")
            ax2.set_ylim((0, 1))
            ax2.legend()
            fig.canvas.draw()

            #image = np.array(pltfig.canvas.renderer._renderer)
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            cv2.imwrite("probs/probs3/%06d.png" % self.t, data)

            #fig.clf()
            ax1.cla()
            ax2.cla()
            """

            mean = (vs.q_values.data * act_probs).sum(1)
            sigma = (vs.sigmas.data * act_probs).sum(1)

            #for i in range(n):
            #    diff = ((start + interval*i) - mean)**2.0
            #    sigma += diff*(int_p[i]*interval)

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
