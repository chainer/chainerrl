from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import contextlib
import os
import random

import chainer
import numpy as np


def set_random_seed(seed, gpus=()):
    """Set a given random seed to ChainerRL's random sources.

    This function sets a given random seed to random sources that ChainerRL
    depends on so that ChainerRL can be deterministic. It is not responsible
    for setting a random seed to environments ChainerRL is applied to.

    Note that there's no guaranteed way to make all the computations done by
    Chainer deterministic. See https://github.com/chainer/chainer/issues/4134.

    Args:
        seed (int): Random seed [0, 2 ** 32).
        gpus (tuple of ints): GPU device IDs to use. Negative values are
            ignored.
    """
    # ChainerRL depends on random
    random.seed(seed)
    # ChainerRL depends on numpy.random
    np.random.seed(seed)
    # ChainerRL depends on cupy.random for GPU computation
    for gpu in gpus:
        if gpu >= 0:
            with chainer.cuda.get_device_from_id(gpu):
                chainer.cuda.cupy.random.seed(seed)
    # chainer.functions.n_step_rnn directly depends on CHAINER_SEED
    os.environ['CHAINER_SEED'] = str(seed)


@contextlib.contextmanager
def using_numpy_random_for_gym_spaces():
    from gym import spaces
    gym_spaces_random_state = spaces.prng.np_random
    spaces.prng.np_random = np.random.rand.__self__
    yield
    spaces.prng.np_random = gym_spaces_random_state


def sample_from_space(space):
    """Sample from gym.spaces.Space.

    Unlike gym.spaces.Space.sample, this function use numpy's global random
    state.

    Users should use this function instead of gym.spaces.Space.sample because
    it is not recommended to use gym.space.Space.sample in algorithms.
    See https://github.com/openai/gym/blob/master/gym/spaces/prng.py

    Args:
        space (gym.spaces.Space): Space.

    Returns:
        object: Sample from the given space.
    """
    with using_numpy_random_for_gym_spaces():
        return space.sample()
