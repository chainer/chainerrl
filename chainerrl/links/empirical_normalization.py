import chainer
import numpy as np


class EmpiricalNormalization(chainer.Link):
    """Normalize mean and variance of values based on emprical values.

    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
    """

    def __init__(self, shape, batch_axis=0, eps=1e-2, dtype=np.float32,
                 until=None):
        super(EmpiricalNormalization, self).__init__()
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.mean = np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)
        self.var = np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)
        self.count = 0
        self.register_persistent('mean')
        self.register_persistent('var')
        self.register_persistent('count')

        # cache
        self._std_inverse = None

    @property
    def std_inverse(self):
        if self._std_inverse is None:
            # TODO:
            # if self.ddof:
            #     correction = 1. - 1. / self.count

            self._std_inverse = (self.var + self.eps) ** -0.5

        return self._std_inverse

    def experience(self, x):
        if self.until is not None and self.count >= self.until:
            return

        if isinstance(x, chainer.Variable):
            x = x.data

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        xp = self.xp

        self.count += count_x
        rate = x.dtype.type(count_x / self.count)

        mean_x = xp.mean(x, axis=self.batch_axis, keepdims=True)
        var_x = xp.var(x, axis=self.batch_axis, keepdims=True)
        delta_mean = mean_x - self.mean
        self.mean += rate * delta_mean
        self.var += rate * (
            var_x - self.var
            + delta_mean * (mean_x - self.mean) 
        )

        # clear cache
        self._std_inverse = None

    def __call__(self, x, update=True):
        xp = self.xp
        mean = xp.broadcast_to(self.mean, x.shape)
        std_inv = xp.broadcast_to(self.std_inverse, x.shape)

        if update:
            self.experience(x)

        return (x - mean) * std_inv

    def inverse(self, x):
        xp = self.xp
        mean = xp.broadcast_to(self.mean, x.shape)
        std = xp.broadcast_to(xp.sqrt(self.var + self.eps), x.shape)
        return x * std + mean
