# This caused an error in py2 because cupy expect non-unicode str
# from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
from chainer import cuda


class NonbiasWeightDecay(object):

    """Optimizer hook function for weight decay regularization.

    """
    name = 'NonbiasWeightDecay'

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, opt):
        if cuda.available:
            kernel = cuda.elementwise(
                'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')

        rate = self.rate
        for name, param in opt.target.namedparams():
            if name == 'b' or name.endswith('/b'):
                continue
            p, g = param.data, param.grad
            with cuda.get_device(p) as dev:
                if int(dev) == -1:
                    g += rate * p
                else:
                    kernel(p, rate, g)
