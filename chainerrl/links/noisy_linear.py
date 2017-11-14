import chainer
import chainer.functions as F
from chainer.initializers import Constant
import chainer.links as L

from chainerrl.initializers import VarianceScalingConstant


class FactorizedNoisyLinear(chainer.Chain):
    """Linear layer in Factorized Noisy Network

    Args:
        in_size, out_size, nobias, initialW, initial_bias: args for L.Linear
        sigma_scale (float): The hyperparameter sigma_0 in the original paper.
            Scaling factor of the initial weights of noise-scaling parameters.
    """

    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None,
                 sigma_scale=0.4):
        super(FactorizedNoisyLinear, self).__init__()
        self.nobias = nobias
        with self.init_scope():
            self.mu = L.Linear(
                in_size, out_size, nobias, initialW, initial_bias)
            self.sigma = L.Linear(
                in_size, out_size, nobias,
                initialW=VarianceScalingConstant(sigma_scale),
                initial_bias=Constant(sigma_scale))

    def _eps(self, shape, dtype):
        xp = self.xp
        r = xp.random.standard_normal(shape).astype(dtype)

        # apply the function f
        return xp.copysign(xp.sqrt(xp.abs(r)), r)

    def __call__(self, x):
        if self.mu.W.data is None:
            # initialize self.mu.W
            self.mu(x[:0])
        if self.sigma.W.data is None:
            # initialize self.sigma.W
            self.sigma(x[:0])

        # use info of sigma.W to avoid strange error messages
        dtype = self.sigma.W.dtype
        out_size, in_size = self.sigma.W.shape

        eps_x = self._eps(in_size, dtype)
        eps_y = self._eps(out_size, dtype)
        W = self.mu.W + self.sigma.W * self.xp.outer(eps_y, eps_x)
        if self.nobias:
            return F.linear(x, W)
        else:
            b = self.mu.b + self.sigma.b * eps_y
            return F.linear(x, W, b)
