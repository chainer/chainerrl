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


class SoftmaxPolicy(Policy):
    """Abstract softmax policy class.
    """

    def forward(self, state):
        """
        Returns:
          ~chainer.Variable: logits of actions
        """
        raise NotImplementedError

    def __call__(self, state, action):
        assert state.shape[0] == 1
        xp = cuda.get_array_module(state)
        logits = self.forward(state)
        probs = F.softmax(logits)
        q = F.select_item(
            probs, chainer.Variable(xp.asarray(action, dtype=np.int32)))
        return q

    def sample_with_probability(self, state):
        """
        Returns:
          ~list: action indices
          ~chainer.Variable: probabilities of sampled actions
        """
        logits = self.forward(state)
        probs = F.softmax(logits)
        action_indices = _sample_actions(probs.data)
        sampled_actions_probs = F.select_item(
            probs,
            chainer.Variable(np.asarray(action_indices, dtype=np.int32)))
        return action_indices, sampled_actions_probs

    def sample_with_log_probability(self, state):
        """
        Returns:
          ~list: action indices
          ~chainer.Variable: log probabilities of sampled actions
        """
        logits = self.forward(state)
        probs = F.softmax(logits)
        action_indices = _sample_actions(probs.data)
        log_probs = F.log_softmax(logits)
        sampled_actions_log_probs = F.select_item(
            log_probs,
            chainer.Variable(np.asarray(action_indices, dtype=np.int32)))
        return action_indices, sampled_actions_log_probs

    def sample_with_log_probability_and_entropy(self, state):
        """
        Returns:
          ~list: action indices
          ~chainer.Variable: log probabilities of sampled actions
        """
        logits = self.forward(state)
        probs = F.softmax(logits)
        action_indices = _sample_actions(probs.data)
        log_probs = F.log_softmax(logits)
        sampled_actions_log_probs = F.select_item(
            log_probs,
            chainer.Variable(np.asarray(action_indices, dtype=np.int32)))
        # Entropy
        entropy = - F.sum(probs * log_probs, axis=1)
        # TODO Too many return values; must re-consider interfaces
        return action_indices, sampled_actions_log_probs, entropy, probs


class FCSoftmaxPolicy(chainer.ChainList, SoftmaxPolicy):

    def __init__(self, n_input_channels, n_actions, n_hidden_channels,
                 n_hidden_layers):
        self.n_input_channels = n_input_channels
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        assert n_hidden_layers >= 1
        layers.append(L.Linear(n_input_channels, n_hidden_channels))
        for i in range(n_hidden_layers - 1):
            layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
        layers.append(L.Linear(n_hidden_channels, n_actions))

        super(FCSoftmaxPolicy, self).__init__(*layers)

    def forward(self, state):
        h = chainer.Variable(state)
        for layer in self[:-1]:
            h = F.elu(layer(h))
        h = self[-1](h)
        return h


class GaussianPolicy(Policy):
    """Abstract gaussian policy class.
    """

    def forward(self, state):
        raise NotImplementedError

    def sample_with_probability(self, state):
        raise NotImplementedError
