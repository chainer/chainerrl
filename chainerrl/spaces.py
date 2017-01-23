# This module is just an alias to gym.spaces except that you don't need to
# call gym.undo_logger_setup().
import gym
gym.undo_logger_setup()
from gym.spaces import *  # NOQA
