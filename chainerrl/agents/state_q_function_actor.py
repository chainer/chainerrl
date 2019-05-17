from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from logging import getLogger

import chainer
from chainer import cuda

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states


class StateQFunctionActor(agent.AsyncAgent):
    """Actor that acts according to the Q-function."""

    process_idx = None
    shared_attributes = []

    def __init__(
        self,
        pipe,
        model,
        explorer,
        phi=lambda x: x,
        recurrent=False,
        logger=getLogger(__name__),
        batch_states=batch_states,
    ):
        self.pipe = pipe
        self.model = model
        self.explorer = explorer
        self.phi = phi
        self.recurrent = recurrent
        self.logger = logger
        self.batch_states = batch_states

        self.t = 0
        self.last_state = None
        self.last_action = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

    @property
    def xp(self):
        return self.model.xp

    def _evaluate_model_and_update_train_recurrent_states(self, batch_obs):
        batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
        if self.recurrent:
            self.train_prev_recurrent_states = self.train_recurrent_states
            batch_av, self.train_recurrent_states = self.model(
                batch_xs, self.train_recurrent_states)
        else:
            batch_av = self.model(batch_xs)
        return batch_av

    def _evaluate_model_and_update_test_recurrent_states(self, batch_obs):
        batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
        if self.recurrent:
            batch_av, self.test_recurrent_states = self.model(
                batch_xs, self.test_recurrent_states)
        else:
            batch_av = self.model(batch_xs)
        return batch_av

    def act(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value =\
                self._evaluate_model_and_update_test_recurrent_states([obs])
            action = cuda.to_cpu(action_value.greedy_actions.array)[0]
        return action

    def _send_to_learner(self, transition, stop_episode=False):
        self.pipe.send(('transition', transition))
        if stop_episode:
            self.pipe.send(('stop_episode', None))

    def act_and_train(self, obs, reward):

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value =\
                self._evaluate_model_and_update_train_recurrent_states([obs])
            greedy_action = cuda.to_cpu(action_value.greedy_actions.array)[0]

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        self.t += 1

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            transition = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': obs,
                'is_state_terminal': False,
            }
            if self.recurrent:
                transition['recurrent_state'] =\
                    self.model.get_recurrent_state_at(
                        self.train_prev_recurrent_states,
                        0, unwrap_variable=True)
                self.train_prev_recurrent_states = None
                transition['next_recurrent_state'] =\
                    self.model.get_recurrent_state_at(
                        self.train_recurrent_states, 0, unwrap_variable=True)
            self._send_to_learner(transition)

        self.last_state = obs
        self.last_action = action

        return self.last_action

    def stop_episode_and_train(self, state, reward, done=False):

        assert self.last_state is not None
        assert self.last_action is not None

        # Add a transition to the replay buffer
        transition = {
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'next_state': state,
            'is_state_terminal': done,
        }
        if self.recurrent:
            transition['recurrent_state'] =\
                self.model.get_recurrent_state_at(
                    self.train_prev_recurrent_states, 0, unwrap_variable=True)
            self.train_prev_recurrent_states = None
            transition['next_recurrent_state'] =\
                self.model.get_recurrent_state_at(
                    self.train_recurrent_states, 0, unwrap_variable=True)
        self._send_to_learner(transition, stop_episode=True)

        self.last_state = None
        self.last_action = None
        if self.recurrent:
            self.train_recurrent_states = None

    def stop_episode(self):
        if self.recurrent:
            self.test_recurrent_states = None

    def save(self, dirname):
        self.pipe.send(('save', dirname))
        self.pipe.recv()

    def load(self, dirname):
        self.pipe.send(('load', dirname))
        self.pipe.recv()

    def get_statistics(self):
        self.pipe.send(('get_statistics', None))
        return self.pipe.recv()
