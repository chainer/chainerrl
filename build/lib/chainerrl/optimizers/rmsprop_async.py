from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

from chainer import cuda
from chainer import optimizer
import numpy


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.alpha = 0.99
_default_hyperparam.eps = 1e-8


class RMSpropAsyncRule(optimizer.UpdateRule):

    def __init__(self, parent_hyperparam=None, lr=None, alpha=None, eps=None):
        super(RMSpropAsyncRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if eps is not None:
            self.hyperparam.eps = eps

    def init_state(self, param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['ms'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        ms = self.state['ms']

        ms *= hp.alpha
        ms += (1 - hp.alpha) * grad * grad
        param.data -= hp.lr * grad / numpy.sqrt(ms + hp.eps)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        cuda.elementwise(
            'T grad, T lr, T alpha, T eps',
            'T param, T ms',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               param -= lr * grad / sqrt(ms + eps);''',
            'rmsprop')(grad, self.hyperparam.lr, self.hyperparam.alpha,
                       self.hyperparam.eps, param.data, self.state['ms'])


class RMSpropAsync(optimizer.GradientMethod):

    """RMSprop for asynchronous methods.

    The only difference from chainer.optimizers.RMSprop in that the epsilon is
    outside the square root.
    """

    def __init__(self, lr=_default_hyperparam.lr,
                 alpha=_default_hyperparam.alpha, eps=_default_hyperparam.eps):
        super(RMSpropAsync, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.alpha = alpha
        self.hyperparam.eps = eps

    lr = optimizer.HyperparameterProxy('lr')
    alpha = optimizer.HyperparameterProxy('alpha')
    eps = optimizer.HyperparameterProxy('eps')

    def create_update_rule(self):
        return RMSpropAsyncRule(self.hyperparam)
