from chainer import cuda
from chainer import optimizers


class RMSpropOnes(optimizers.RMSprop):
    """RMSprop with initialization with ones

    This is the same as chainer.optimizers.RMSprop except it uses ones for
    initialize mean square gradients.
    """

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        state['ms'] = xp.ones_like(param.data)
