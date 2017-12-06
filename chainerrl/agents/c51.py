from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import chainer
import chainer.cuda as cuda
import chainer.functions as F

from chainerrl.agents import dqn
from chainerrl.recurrent import state_kept

import math

import cv2
import numpy as np

class C51(dqn.DQN):
    """Value distribution algorithm.
    """

    def _compute_target_values(self, exp_batch, gamma):
        batch_next_state = exp_batch['next_state']

        # (batch_size, n_actions, n_atoms)
        target_next_qout = self.target_model(batch_next_state)
        next_q_max = target_next_qout.max_distribution.data

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        batch_size = exp_batch['reward'].shape[0]
        n_atoms = next_q_max.shape[1]
        target_values = self.xp.zeros((batch_size, n_atoms), dtype=self.xp.float32)
        action = exp_batch['action']

        # TODO
        """
        
        """
        #a_i = action[i]

        for j in range(n_atoms):
            Tz = self.xp.clip(exp_batch['reward'] + (1.0 - batch_terminal) * gamma * self.z_values[j], self.v_min, self.v_max)
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = self.xp.floor(bj), self.xp.ceil(bj)
            target_values[self.xp.arange(batch_size), m_l.astype(self.xp.int16)] += next_q_max[:, j] * (m_u - bj)
            target_values[self.xp.arange(batch_size), m_u.astype(self.xp.int16)] += next_q_max[:, j] * (bj - m_l)

        #canvas = np.zeros([480, 640, 3])

        def draw_dist(canvas, dist, offset_x, offset_y, col):
            w = 10
            for i, p in enumerate(dist):
                canvas = cv2.rectangle(canvas, (offset_x + i*w, offset_y), (offset_x + (i+1)*w, offset_y-p*1000), col, -1)
            return canvas

        #canvas = draw_dist(canvas, next_q_max[0], 50, 200, (255,0,0))
        #canvas = draw_dist(canvas, target_values[0], 50, 400, (0,255,0))
        #canvas = cv2.putText(canvas, str(action[0]), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        #cv2.imshow("update", canvas)
        #cv2.waitKey(1)

        return target_values

    def _compute_y_and_t(self, exp_batch, gamma):
        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        # (batch_size, n_actions, n_atoms)
        qout = self.model(batch_state)

        batch_actions = exp_batch['action']
        h = qout.evaluate_actions(
            batch_actions)
        batch_q = F.reshape(h, (batch_size, self.n_atoms))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch, gamma),
                (batch_size, self.n_atoms))

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, gamma, errors_out=None):
        """Compute the Q-learning loss for a batch of experiences


        Args:
          experiences (list): see update()'s docstring
          gamma (float): discount factor
        Returns:
          loss
        """

        self.n_atoms = 51
        self.v_min = 0#-10
        self.v_max = 500#10
        self.delta_z = (self.v_max - self.v_min) / float(self.n_atoms - 1)
        self.z_values = self.xp.array([self.v_min + i * self.delta_z for i in range(self.n_atoms)])

        y, t = self._compute_y_and_t(exp_batch, gamma)
        #print(y, t)

        canvas = np.zeros([480, 640, 3])

        def draw_dist(canvas, dist, offset_x, offset_y, col):
            w = 10
            for i, p in enumerate(dist):
                canvas = cv2.rectangle(canvas, (offset_x + i*w, offset_y), (offset_x + (i+1)*w, offset_y-p*1000), col, -1)
            return canvas

        """
        if errors_out is not None:
            del errors_out[:]
            delta = F.sum(abs(y - t), axis=1)
            delta = cuda.to_cpu(delta.data)
            for e in delta:
                errors_out.append(e)
        """

        #return compute_value_loss(y, t, clip_delta=self.clip_delta,
        #                              batch_accumulator=self.batch_accumulator)
        #t = self.xp.zeros([exp_batch['reward'].shape[0], self.n_atoms])
        #t[:, 0] = 1

        #y = self.model.model(exp_batch['state'])[self.xp.arange(t.shape[0]), 0, :]

        if self.t % 1 == 0:
            canvas = draw_dist(canvas, y.data[0], 50, 200, (255,0,0))
            canvas = draw_dist(canvas, t.data[0], 50, 400, (0,255,0))
            cv2.imshow("dists", canvas)
            cv2.waitKey(1)

        #print(t.shape, F.sum(t, axis=1))
        #print(y.shape, F.sum(y, axis=1))
        #print(F.log(y[0]+1e-6))
        loss = -F.mean(F.sum(t * F.log(y+1e-5), axis=1))
        #print(loss)
        #print(np.mean(chainer.grad([loss], [self.model.model.outputs[0].W])[0].data))

        return loss
