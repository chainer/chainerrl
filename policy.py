import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L


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
    for i in xrange(batch_probs.shape[0]):
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
            softmax_probs,
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
        for i in xrange(n_hidden_layers - 1):
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
