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


def test():
    en = EmpiricalNormalization(10)
    xs = []
    for _ in range(10000):
        x = np.random.normal(loc=4, scale=2, size=(7, 10))
        en(x)
        xs.extend(list(x))
    xs = np.array(xs)
    true_mean = np.mean(xs, axis=0, keepdims=True)
    true_std = np.std(xs, axis=0, keepdims=True)
    np.testing.assert_allclose(en.mean, true_mean, rtol=1e-4)
    np.testing.assert_allclose(np.sqrt(en.var), true_std, rtol=1e-4)

    x = 2 * np.random.normal(loc=4, scale=2, size=(1, 10))
    enx = en(x)
    # mean, std = en.mean_and_std()
    mean = en.mean
    std = np.sqrt(en.var)
    print('mean', mean)
    np.testing.assert_allclose(mean, np.full_like(mean, 4), rtol=1e-1)
    print('std', std)
    np.testing.assert_allclose(std, np.full_like(std, 2), rtol=1e-1)
    print('ground-truth normaliaztion', (x - 4) / 2)
    print('en(x)', enx)
    np.testing.assert_allclose((x - 4) / 2, enx, rtol=1e-1)
    np.testing.assert_allclose(x, en.inverse(enx), rtol=1e-1)


if __name__ == '__main__':
    test()
