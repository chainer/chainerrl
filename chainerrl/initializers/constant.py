from chainer import initializer
from chainer.initializers import Constant
import numpy


class VarianceScalingConstant(initializer.Initializer):
    def __init__(self, scale=1.0, fan='in', dtype=None):
        super(VarianceScalingConstant, self).__init__(dtype)
        self.scale = scale
        self.fan = fan

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype

        if len(array.shape) == 1:
            Constant(self.scale / numpy.sqrt(array.shape[0]))(array)
        else:
            fan_in, fan_out = initializer.get_fans(array.shape)
            
            if self.fan == 'out':
                Constant(self.scale / numpy.sqrt(fan_out))(array)
            else:
                Constant(self.scale / numpy.sqrt(fan_in))(array)
