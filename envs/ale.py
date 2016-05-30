import collections
import os
import sys

import numpy as np
from ale_python_interface import ALEInterface
import cv2

import environment


class ALE(environment.EpisodicEnvironment):
    """Arcade Learning Environment.
    """

    def __init__(self, rom_filename, seed=None, use_sdl=False, n_last_screens=4,
                 frame_skip=4, treat_life_lost_as_terminal=True,
                 crop_or_scale='scale', max_start_nullops=30,
                 record_screen_dir=None):
        self.n_last_screens = n_last_screens
        self.treat_life_lost_as_terminal = treat_life_lost_as_terminal
        self.crop_or_scale = crop_or_scale
        self.max_start_nullops = max_start_nullops

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
            ale.setString(b'record_screen_dir', str.encode(record_screen_dir))
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
        ale.loadROM(str.encode(rom_filename))

        assert ale.getFrameNumber() == 0


        self.ale = ale
        self.legal_actions = ale.getMinimalActionSet()
        self.initialize()

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
            img = cv2.resize(img, (84, 110),
                             interpolation=cv2.INTER_LINEAR)
            assert img.shape == (110, 84)
            # Crop (110, 84) -> (84, 84)
            unused_height = 110 - 84
            bottom_crop = 8
            top_crop = unused_height - bottom_crop
            img = img[top_crop: 110 - bottom_crop, :]
        elif self.crop_or_scale == 'scale':
            img = cv2.resize(img, (84, 84),
                             interpolation=cv2.INTER_LINEAR)
        else:
            raise RuntimeError('crop_or_scale must be either crop or scale')
        assert img.shape == (84, 84)
        return img

    @property
    def state(self):
        assert len(self.last_screens) == 4
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
