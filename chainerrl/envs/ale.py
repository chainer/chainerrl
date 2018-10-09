from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import collections
import os
import sys
import warnings

from gym import spaces
import numpy as np

from chainerrl import env


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
    from distutils.version import StrictVersion
    import pkg_resources
    atari_py_version = StrictVersion(
        pkg_resources.get_distribution("atari_py").version)
    if atari_py_version < StrictVersion('0.1.1'):
        warnings.warn(
            'atari_py is old. You need to install atari_py>=0.1.1 to use ALE.')  # NOQA
        atari_py_available = False
    else:
        atari_py_available = True
except Exception:
    atari_py_available = False
    warnings.warn(
        'atari_py is not available. You need to install atari_py>=0.1.1 to use ALE.')  # NOQA


class ALE(env.Env):
    """The Arcade Learning Environment with popular settings.

    This mimics the environments used by the DQN paper from DeepMind,
    https://www.nature.com/articles/nature14236.

    Args:
        game (str): Name of a game. You can get the complete list of supported
            games by calling atari_py.list_games().
        seed (int or None): If set to an int, it is used as a random seed for
            the ALE. It must be in [0, 2 ** 31). If set to None, numpy's random
            state is used to select a random seed for the ALE.
        use_sdl (bool): If set to True, use SDL to show a window to render
            states. This option might not work for the ALE in atari_py.
        n_last_screens (int): Number of last screens to observe every step.
        frame_skip (int): Number of frames for which the same action is
            repeated. For example, if it is set to 4, one step for the agent
            corresponds to four frames in the ALE.
        crop_or_scale (str): How screens are resized. If set to 'crop', screens
            are cropped as in https://arxiv.org/abs/1312.5602. If set to
            'scale', screens are scaled as in
            https://www.nature.com/articles/nature14236.
        max_start_nullops (int): Maximum number of random null actions sent to
            the ALE to randomize initial states.
        record_screen_dir (str): If set to a str, screens are saved as images
            to the directory specified by it. If set to None, screens are not
            saved.
    """

    def __init__(self, game, seed=None, use_sdl=False, n_last_screens=4,
                 frame_skip=4, treat_life_lost_as_terminal=True,
                 crop_or_scale='scale', max_start_nullops=30,
                 record_screen_dir=None):
        assert crop_or_scale in ['crop', 'scale']
        assert frame_skip >= 1
        self.n_last_screens = n_last_screens
        self.treat_life_lost_as_terminal = treat_life_lost_as_terminal
        self.crop_or_scale = crop_or_scale
        self.max_start_nullops = max_start_nullops

        # atari_py is used only to provide rom files. atari_py has its own
        # ale_python_interface, but it is obsolete.
        if not atari_py_available:
            raise RuntimeError(
                'You need to install atari_py>=0.1.1 to use ALE.')
        game_path = atari_py.get_game_path(game)

        ale = atari_py.ALEInterface()
        if seed is not None:
            assert seed >= 0 and seed < 2 ** 31, \
                "ALE's random seed must be in [0, 2 ** 31)."
        else:
            # Use numpy's random state
            seed = np.random.randint(0, 2 ** 31)
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
            low=0, high=255,
            shape=(84, 84), dtype=np.uint8,
        )
        self.observation_space = spaces.Tuple(
            [one_screen_observation_space] * n_last_screens)

    def current_screen(self):
        # Max of two consecutive frames
        assert self.last_raw_screen is not None
        rgb_img = np.maximum(self.ale.getScreenRGB(), self.last_raw_screen)
        # Make sure the last raw screen is used only once
        self.last_raw_screen = None
        assert rgb_img.shape[2:] == (3,)
        # RGB -> Luminance
        img = np.dot(rgb_img, np.array([0.299, 0.587, 0.114]))
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
        for i in range(self.frame_skip):

            # Last screeen must be stored before executing the last action
            if i == self.frame_skip - 1:
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
            [np.zeros((84, 84), dtype=np.uint8)] * (self.n_last_screens - 1) +
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
