# This caused an error in py2 because cupy expect non-unicode str
# from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
from chainer import cuda


class NonbiasWeightDecay(object):

    """Weight decay only for non-bias parameters.

    This hook can be used just like chainer.optimizer_hooks.WeightDecay except
    that this hook does not apply weight decay to bias parameters.

    This hook assumes that all the bias parameters have the name of "b". Any
    parameter whose name is "b" is considered as a bias and excluded from
    weight decay.
    """
    name = 'NonbiasWeightDecay'
    call_for_each_param = True
    timing = 'pre'

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        if param.name == 'b':
            return
        p, g = param.array, param.grad
        if p is None or g is None:
            return
        with cuda.get_device_from_array(p) as dev:
            if int(dev) == -1:
                g += self.rate * p
            else:
                kernel = cuda.elementwise(
                    'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')
                kernel(p, self.rate, g)
