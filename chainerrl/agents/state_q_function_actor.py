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
        queue,
        pipe,
        model,
        explorer,
        phi=lambda x: x,
        logger=getLogger(__name__),
        batch_states=batch_states,
    ):
        self.queue = queue
        self.pipe = pipe
        self.model = model
        self.explorer = explorer
        self.phi = phi
        self.logger = logger
        self.batch_states = batch_states

        self.t = 0
        self.last_state = None
        self.last_state = None
        self.batch_last_action = None
        self.batch_last_action = None

    @property
    def xp(self):
        return self.model.xp

    def compute_action_value(self, batch_obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return self.model(self.batch_states(batch_obs, self.xp, self.phi))

    def act(self, obs):
        action_value = self.compute_action_value([obs])
        action = cuda.to_cpu(action_value.greedy_actions.array)[0]
        return action

    def _send_to_learner(self, transition, stop_episode=False):
        self.queue.put((self.process_idx, 'transition', transition))
        if stop_episode:
            self.queue.put((self.process_idx, 'stop_episode', None))

    def act_and_train(self, obs, reward):

        action_value = self.compute_action_value([obs])
        greedy_action = cuda.to_cpu(action_value.greedy_actions.array)[0]

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        self.t += 1

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self._send_to_learner(dict(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                next_action=action,
                is_state_terminal=False),
                stop_episode=False,
            )

        self.last_state = obs
        self.last_action = action

        return self.last_action

    def stop_episode_and_train(self, state, reward, done=False):

        assert self.last_state is not None
        assert self.last_action is not None

        # Add a transition to the replay buffer
        self._send_to_learner(dict(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=state,
            next_action=self.last_action,
            is_state_terminal=done),
            stop_episode=True,
        )

        self.last_state = None
        self.last_action = None

    def stop_episode(self):
        pass

    def save(self, dirname):
        self.pipe.send((self.process_idx, 'save', dirname))
        self.pipe.recv()

    def load(self, dirname):
        self.pipe.send((self.process_idx, 'load', dirname))
        self.pipe.recv()

    def get_statistics(self):
        self.pipe.send((self.process_idx, 'get_statistics', None))
        return self.pipe.recv()
