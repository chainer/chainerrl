from logging import getLogger
logger = getLogger(__name__)

import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda


class Policy(object):
    """Abstract policy class.
    """

    def sample_with_probability(self, state):
        raise NotImplementedError


def _sample_actions(batch_probs):
    """
    Args:
      batch_probs (ndarray): batch of action probabilities BxA
    """
    action_indices = []
    # Avoid "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    batch_probs = batch_probs - np.finfo(np.float32).epsneg
    for i in range(batch_probs.shape[0]):
        histogram = np.random.multinomial(1, batch_probs[i])
        action_indices.append(int(np.nonzero(histogram)[0]))
    return action_indices


class PolicyOutput(object):
    pass


class SoftmaxPolicyOutput(PolicyOutput):

    def __init__(self, logits):
        self.logits = logits
        self.probs = F.softmax(logits)
        self.action_indices = _sample_actions(self.probs.data)
        self.log_probs = F.log_softmax(logits)
        self.sampled_actions_log_probs = F.select_item(
            self.log_probs,
            chainer.Variable(np.asarray(self.action_indices, dtype=np.int32)))
        self.entropy = - F.sum(self.probs * self.log_probs, axis=1)


class SoftmaxPolicy(Policy):
    """Abstract softmax policy class.
    """

    def compute_logits(self, state):
        """
        Returns:
          ~chainer.Variable: logits of actions
        """
        raise NotImplementedError

    def __call__(self, state):
        return SoftmaxPolicyOutput(self.compute_logits(state))


class FCSoftmaxPolicy(chainer.ChainList, SoftmaxPolicy):

    def __init__(self, n_input_channels, n_actions,
                 n_hidden_layers=0, n_hidden_channels=None):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        if n_hidden_layers > 0:
            layers.append(L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
            layers.append(L.Linear(n_hidden_channels, n_actions))
        else:
            layers.append(L.Linear(n_input_channels, n_actions))

        super(FCSoftmaxPolicy, self).__init__(*layers)

    def compute_logits(self, state):
        h = state
        for layer in self[:-1]:
            h = F.relu(layer(h))
        h = self[-1](h)
        return h


class GaussianPolicy(Policy):
    """Abstract gaussian policy class.
    """

    def forward(self, state):
        raise NotImplementedError

    def sample_with_probability(self, state):
        raise NotImplementedError
