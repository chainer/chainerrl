from chainer import cuda
from chainer import function
from chainer.utils import type_check


class SmoothL1Loss(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1 = inputs
        self.diff = x0 - x1
        y = (0.5 * xp.square(self.diff) * (xp.abs(self.diff) < 1) +
             (xp.abs(self.diff) - 0.5) * (xp.abs(self.diff) >= 1))
        return y.sum(axis=1),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        gx = (self.diff * (xp.abs(self.diff) < 1) +
              xp.sign(self.diff) * (xp.abs(self.diff) >= 1))
        return gx, -gx


def smooth_l1_loss(x, t):
    return SmoothL1Loss()(x, t)

