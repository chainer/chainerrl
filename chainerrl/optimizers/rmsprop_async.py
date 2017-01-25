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


class RMSpropAsync(optimizer.GradientMethod):

    """RMSprop for asynchronous methods.

    The only difference from chainer.optimizers.RMSprop in that the epsilon is
    outside the square root.
    """

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        state['ms'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        ms = state['ms']
        grad = param.grad

        ms *= self.alpha
        ms += (1 - self.alpha) * grad * grad
        param.data -= self.lr * grad / numpy.sqrt(ms + self.eps)

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T alpha, T eps',
            'T param, T ms',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               param -= lr * grad / sqrt(ms + eps);''',
            'rmsprop')(param.grad, self.lr, self.alpha, self.eps,
                       param.data, state['ms'])
