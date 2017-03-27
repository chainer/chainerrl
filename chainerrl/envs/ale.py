from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import collections
import os
import sys
import warnings

from ale_python_interface import ALEInterface
import numpy as np

from chainerrl import env
from chainerrl import spaces


try:
    import cv2

    def imresize(img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

except Exception:
    from PIL import Image

    warnings.warn(
        'Since cv2 is not available PIL will be used instead to resize images.'
        ' This might affect the resulting images.')

    def imresize(img, size):
        return np.asarray(Image.fromarray(img).resize(size, Image.BILINEAR))

try:
    import atari_py
    atari_py_available = True
except Exception:
    atari_py_available = False
    warnings.warn(
        'atari_py is not available. You need to install atari_py to use ALE.')


class ALE(env.Env):
    """Arcade Learning Environment."""

    def __init__(self, game, seed=None, use_sdl=False, n_last_screens=4,
                 frame_skip=4, treat_life_lost_as_terminal=True,
                 crop_or_scale='scale', max_start_nullops=30,
                 record_screen_dir=None):
        self.n_last_screens = n_last_screens
        self.treat_life_lost_as_terminal = treat_life_lost_as_terminal
        self.crop_or_scale = crop_or_scale
        self.max_start_nullops = max_start_nullops

        # atari_py is used only to provide rom files. atari_py has its own
        # ale_python_interface, but it is obsolete.
        if not atari_py_available:
            raise RuntimeError('You need to install atari_py to use ALE.')
        game_path = atari_py.get_game_path(game)

        ale = ALEInterface()
        if seed is not None:
            assert seed >= 0 and seed < 2 ** 16, \
                "ALE's random seed must be represented by unsigned int"
        else:
            # Use numpy's random state
            seed = np.random.randint(0, 2 ** 16)
        ale.setInt(b'random_seed', seed)
        ale.setFloat(b'repeat_action_probability', 0.0)
        ale.setBool(b'color_averaging', False)
        if record_screen_dir is not None:
            ale.setString(b'record_screen_dir',
                          str.encode(str(record_screen_dir)))
        self.frame_skip = frame_skip
        if use_sdl:
            if 'DISPLAY' not in os.environ:
                raise RuntimeError(
                    'Please set DISPLAY environment variable for use_sdl=True')
            # SDL settings below are from the ALE python example
            if sys.platform == 'darwin':
                import pygame
                pygame.init()
                ale.setBool(b'sound', False)  # Sound doesn't work on OSX
            elif sys.platform.startswith('linux'):
                ale.setBool(b'sound', True)
            ale.setBool(b'display_screen', True)

        ale.loadROM(str.encode(str(game_path)))

        assert ale.getFrameNumber() == 0

        self.ale = ale
        self.legal_actions = ale.getMinimalActionSet()
        self.initialize()

        self.action_space = spaces.Discrete(len(self.legal_actions))
        one_screen_observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84))
        self.observation_space = spaces.Tuple(
            [one_screen_observation_space] * n_last_screens)

    def current_screen(self):
        # Max of two consecutive frames
        assert self.last_raw_screen is not None
        rgb_img = np.maximum(self.ale.getScreenRGB(), self.last_raw_screen)
        # Make sure the last raw screen is used only once
        self.last_raw_screen = None
        assert rgb_img.shape == (210, 160, 3)
        # RGB -> Luminance
        img = rgb_img[:, :, 0] * 0.2126 + rgb_img[:, :, 1] * \
            0.0722 + rgb_img[:, :, 2] * 0.7152
        img = img.astype(np.uint8)
        if img.shape == (250, 160):
            raise RuntimeError("This ROM is for PAL. Please use ROMs for NTSC")
        assert img.shape == (210, 160)
        if self.crop_or_scale == 'crop':
            # Shrink (210, 160) -> (110, 84)
            img = imresize(img, (84, 110))
            assert img.shape == (110, 84)
            # Crop (110, 84) -> (84, 84)
            unused_height = 110 - 84
            bottom_crop = 8
            top_crop = unused_height - bottom_crop
            img = img[top_crop: 110 - bottom_crop, :]
        elif self.crop_or_scale == 'scale':
            img = imresize(img, (84, 84))
        else:
            raise RuntimeError('crop_or_scale must be either crop or scale')
        assert img.shape == (84, 84)
        return img

    @property
    def state(self):
        assert len(self.last_screens) == self.n_last_screens
        return list(self.last_screens)

    @property
    def is_terminal(self):
        if self.treat_life_lost_as_terminal:
            return self.lives_lost or self.ale.game_over()
        else:
            return self.ale.game_over()

    @property
    def reward(self):
        return self._reward

    @property
    def number_of_actions(self):
        return len(self.legal_actions)

    def receive_action(self, action):
        assert not self.is_terminal

        rewards = []
        for i in range(4):

            # Last screeen must be stored before executing the 4th action
            if i == 3:
                self.last_raw_screen = self.ale.getScreenRGB()

            rewards.append(self.ale.act(self.legal_actions[action]))

            # Check if lives are lost
            if self.lives > self.ale.lives():
                self.lives_lost = True
            else:
                self.lives_lost = False
            self.lives = self.ale.lives()

            if self.is_terminal:
                break

        # We must have last screen here unless it's terminal
        if not self.is_terminal:
            self.last_screens.append(self.current_screen())

        self._reward = sum(rewards)

        return self._reward

    def initialize(self):

        if self.ale.game_over():
            self.ale.reset_game()

        if self.max_start_nullops > 0:
            n_nullops = np.random.randint(0, self.max_start_nullops + 1)
            for _ in range(n_nullops):
                self.ale.act(0)

        self._reward = 0

        self.last_raw_screen = self.ale.getScreenRGB()

        self.last_screens = collections.deque(
            [np.zeros((84, 84), dtype=np.uint8)] * 3 +
            [self.current_screen()],
            maxlen=self.n_last_screens)

        self.lives_lost = False
        self.lives = self.ale.lives()

    def reset(self):
        self.initialize()
        return self.state

    def step(self, action):
        self.receive_action(action)
        return self.state, self.reward, self.is_terminal, {}

    def close(self):
        pass
