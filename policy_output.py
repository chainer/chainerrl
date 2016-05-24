import chainer
from chainer import functions as F
from cached_property import cached_property
import numpy as np


class PolicyOutput(object):
    """Struct that holds policy output and subproducts."""

    def entropy(self):
        raise NotImplementedError

    def sampled_actions(self):
        raise NotImplementedError

    def sampled_actions_log_probs(self):
        raise NotImplementedError


def _sample_discrete_actions(batch_probs):
    """Sample a batch of actions from a batch of action probabilities.

    Args:
      batch_probs (ndarray): batch of action probabilities BxA
    Returns:
      List consisting of sampled actions
    """
    action_indices = []

    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    batch_probs = batch_probs - np.finfo(np.float32).epsneg

    for i in range(batch_probs.shape[0]):
        histogram = np.random.multinomial(1, batch_probs[i])
        action_indices.append(int(np.nonzero(histogram)[0]))
    return np.asarray(action_indices, dtype=np.int32)


class SoftmaxPolicyOutput(PolicyOutput):

    def __init__(self, logits):
        self.logits = logits

    @cached_property
    def most_probable_actions(self):
        return chainer.Variable(
            np.argmax(self.probs.data, axis=1).astype(np.int32))

    @cached_property
    def probs(self):
        return F.softmax(self.logits)

    @cached_property
    def log_probs(self):
        return F.log_softmax(self.logits)

    @cached_property
    def action_indices(self):
        return _sample_discrete_actions(self.probs.data)

    @cached_property
    def sampled_actions(self):
        return chainer.Variable(_sample_discrete_actions(self.probs.data))

    @cached_property
    def sampled_actions_log_probs(self):
        return F.select_item(self.log_probs, self.sampled_actions)

    @cached_property
    def entropy(self):
        return - F.sum(self.probs * self.log_probs, axis=1)

    def __repr__(self):
        return 'SoftmaxPolicyOutput logits:{} probs:{} entropy:{}'.format(
            self.logits.data, self.probs.data, self.entropy.data)


class GaussianPolicyOutput(PolicyOutput):

    def __init__(self, mean, ln_var=None, var=None):
        self.mean = mean

        if ln_var is not None:
            assert var is None
            self.ln_var = ln_var
            self.var = F.exp(ln_var)
        elif var is not None:
            assert ln_var is None
            self.ln_var = F.log(var)
            self.var = var

    @cached_property
    def most_probable_actions(self):
        return self.mean

    @cached_property
    def sampled_actions(self):
        return F.gaussian(self.mean, self.ln_var)

    @cached_property
    def sampled_actions_log_probs(self):
        # log N(x|mean,var)
        #   = -0.5log(2pi) - 0.5log(var) - (x - mean)**2 / (2*var)

        # Since this function is intended to be used in REINFORCE-like updates,
        # it won't backpropagate gradients through sampled actions
        sampled_actions = chainer.Variable(self.sampled_actions.data)

        log_probs = -0.5 * np.log(2 * np.pi) - \
            0.5 * self.ln_var - \
            ((sampled_actions - self.mean) ** 2) / (2 * self.var)
        return F.sum(log_probs, axis=1)

    @cached_property
    def entropy(self):
        # Differential entropy of Gaussian is:
        #   0.5 * (log(2 * pi * var) + 1)
        #   = 0.5 * (log(2 * pi) + log var + 1)
        return 0.5 * self.mean.data.shape[1] * (np.log(2 * np.pi) + 1) + \
            0.5 * F.sum(self.ln_var, axis=1)

    def __repr__(self):
        return 'GaussianPolicyOutput mean:{} ln_var:{} entropy:{}'.format(
            self.mean.data, self.ln_var.data, self.entropy.data)
