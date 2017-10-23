import chainer
import chainer.functions as F
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
                 until=-1):
        super(EmpiricalNormalization, self).__init__()
        self.batch_axis = batch_axis
        self.eps = dtype(eps)
        self.until = until
        self.sum = np.zeros(shape, dtype=dtype)
        self.sumsq = np.full(shape, eps, dtype=dtype)
        self.count = np.full(shape, eps, dtype=dtype)
        self.register_persistent('sum')
        self.register_persistent('sumsq')
        self.register_persistent('count')

    def mean_and_std(self):
        xp = self.xp
        mean = self.sum / self.count
        std = xp.sqrt(
            xp.maximum(
                self.sumsq / self.count - mean ** 2,
                self.eps))
        return mean, std

    def __call__(self, x, update_flags=None):

        xp = self.xp
        mean, std = self.mean_and_std()

        mean = xp.broadcast_to(mean, x.shape)
        std = xp.broadcast_to(std, x.shape)

        if (chainer.configuration.config.train
                and (self.until < 0 or self.count.ravel()[0] < self.until)):
            if isinstance(x, chainer.Variable):
                x_data = x.data
            else:
                x_data = x
            if update_flags is not None:
                assert update_flags.shape == (len(x), 1)
                update_flags_b = xp.broadcast_to(update_flags, x_data.shape)
                x_data = update_flags_b * x_data
                self.sum += x_data.sum(axis=self.batch_axis)
                self.sumsq += (x_data ** 2).sum(axis=self.batch_axis)
                self.count += update_flags.sum()
            else:
                self.sum += x_data.sum(axis=self.batch_axis)
                self.sumsq += (x_data ** 2).sum(axis=self.batch_axis)
                self.count += x_data.shape[self.batch_axis]

        return (x - mean) / std

    def inverse(self, x):
        xp = self.xp
        mean, std = self.mean_and_std()

        mean = xp.broadcast_to(mean, x.shape)
        std = xp.broadcast_to(std, x.shape)
        return x * std + mean


def test():
    en = EmpiricalNormalization(10)
    xs = []
    for _ in range(10000):
        x = np.random.normal(loc=4, scale=2, size=(7, 10))
        en(x)
        xs.extend(np.split(x, 7))
    x = 2 * np.random.normal(loc=4, scale=2, size=(1, 10))
    enx = en(x)
    mean, std = en.mean_and_std()
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
