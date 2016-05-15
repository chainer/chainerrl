import numpy as np


def dqn_phi(screens):
    """Phi (feature extractor) of DQN for ALE
    Args:
      screens: List of N screen objects. Each screen object must be
      numpy.ndarray whose dtype is numpy.uint8.
    Returns:
      numpy.ndarray
    """
    assert len(screens) == 4
    assert screens[0].dtype == np.uint8
    raw_values = np.asarray(screens, dtype=np.float32)
    # [0,255] -> [0, 1]
    raw_values /= 255.0
    return raw_values
