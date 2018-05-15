# This module makes spaces in gym backward compatibile.
# Also you don't need to call gym.undo_logger_setup().
try:
    from packaging import version
except ImportError:
    from pip._vendor.packaging import version

import gym
import numpy as np


try:
    gym_version_string = gym.__version__
except AttributeError:
    import pkg_resources
    gym_version_string = pkg_resources.get_distribution("gym").version

gym_version = version.parse(gym_version_string)


def _as_dtype(x, dtype):
    if np.isscalar(x):
        return dtype.type(x)
    else:
        return x.astype(dtype)


if gym_version >= version.parse('0.9.6'):
    from gym.spaces import *  # NOQA
else:
    gym.undo_logger_setup()
    from gym.spaces import *  # NOQA

    class Box(gym.spaces.Box):
        """Box space with the newer (gym>=0.9.6) interface"""

        def __init__(self, low=None, high=None, shape=None, dtype=None):
            if dtype is not None:
                dtype = np.dtype(dtype)
                low = _as_dtype(low, dtype)
                high = _as_dtype(high, dtype)
            super().__init__(dtype.type(low), dtype.type(high), shape)
