import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L


class Policy(object):
    """Abstract policy class.
    """

    def sample_with_probability(self, state):
        raise NotImplementedError


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
        softmax_probs = F.softmax(logits)
        print 'state', state, 'probs', softmax_probs.data
        action_indices = []
        for i in xrange(softmax_probs.data.shape[0]):
            histogram = np.random.multinomial(1, softmax_probs.data[i])
            action_indices.append(int(np.nonzero(histogram)[0]))
        sampled_actions_probs = F.select_item(
            softmax_probs,
            chainer.Variable(np.asarray(action_indices, dtype=np.int32)))
        return action_indices, sampled_actions_probs


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
        print 'forward with state', state
        h = chainer.Variable(state)
        for layer in self[:-1]:
            h = F.relu(layer(h))
            print 'h', h.data
        h = self[-1](h)
        return h


class GaussianPolicy(Policy):
    """Abstract gaussian policy class.
    """

    def forward(self, state):
        raise NotImplementedError

    def sample_with_probability(self, state):
        raise NotImplementedError
