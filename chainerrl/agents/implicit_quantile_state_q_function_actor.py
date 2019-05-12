from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from chainerrl.agents import state_q_function_actor


class ImplicitQuantileStateQFunctionActor(
        state_q_function_actor.StateQFunctionActor):
    """Actor that acts according to the implicit quantile Q-function.

    This actor specialization is required because the interface of an implicit
    quantile Q-function is different from that of a usual Q-function.
    """

    def __init__(self, *args, **kwargs):
        # K=32 were used in the IQN paper's experiments
        # (personal communication)
        self.quantile_thresholds_K = kwargs.pop('quantile_thresholds_K', 32)
        super().__init__(*args, **kwargs)

    @property
    def xp(self):
        return self.model.xp

    def _evaluate_model_and_update_train_recurrent_states(self, batch_obs):
        batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
        if self.recurrent:
            self.train_prev_recurrent_states = self.train_recurrent_states
            tau2av, self.train_recurrent_states = self.model(
                batch_xs, self.train_recurrent_states)
        else:
            tau2av = self.model(batch_xs)
        taus_tilde = self.xp.random.uniform(
            0, 1,
            size=(len(batch_obs), self.quantile_thresholds_K)).astype('f')
        return tau2av(taus_tilde)

    def _evaluate_model_and_update_test_recurrent_states(self, batch_obs):
        batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
        if self.recurrent:
            tau2av, self.test_recurrent_states = self.model(
                batch_xs, self.test_recurrent_states)
        else:
            tau2av = self.model(batch_xs)
        taus_tilde = self.xp.random.uniform(
            0, 1,
            size=(len(batch_obs), self.quantile_thresholds_K)).astype('f')
        return tau2av(taus_tilde)
