import collections
import os
import sys

import numpy as np
import scipy.misc as spm
from ale_python_interface import ALEInterface
from PIL import Image

import environment


class ALE(environment.EpisodicEnvironment):
    """Arcade Learning Environment.
    """

    def __init__(self, rom_filename, seed=0, use_sdl=False, n_last_screens=4,
                 frame_skip=4):
        self.n_last_screens = n_last_screens

        ale = ALEInterface()
        ale.setInt(b'random_seed', seed)
        ale.setInt(b'frame_skip', frame_skip)
        if use_sdl:
            if 'DISPLAY' not in os.environ:
                raise RuntimeError(
                    'Please set DISPLAY environment variable for use_sdl=True')
            # SDL settings below are from the ALE python example
            if sys.platform == 'darwin':
                import pygame
                pygame.init()
                ale.setBool('sound', False)  # Sound doesn't work on OSX
            elif sys.platform.startswith('linux'):
                ale.setBool('sound', True)
            ale.setBool('display_screen', True)
        ale.loadROM(str.encode(rom_filename))

        assert ale.getFrameNumber() == 0

        self.ale = ale
        self.initialize()

    def current_screen(self):
        # 210x160x1
        img = self.ale.getScreenGrayscale()
        if img.shape == (250, 160, 1):
            raise RuntimeError("This ROM is for PAL. Please use ROMs for NTSC")
        assert img.shape == (210, 160, 1)
        # Shrink (210, 160) -> (110, 84)
        img = Image.fromarray(img.reshape(img.shape[:-1]), mode='L')
        assert img.size == (160, 210)
        img = np.asarray(img.resize((84, 110)), dtype=np.float32)
        assert img.shape == (110, 84)
        # Crop (110, 84) -> (84, 84)
        unused_height = 110 - 84
        img = img[unused_height / 2: 110 - unused_height / 2, :]
        assert img.shape == (84, 84)
        # [0,255] -> [-128, 127]
        img -= 128
        # [-128, 127] -> [-1, 1)
        img /= 128
        return img

    @property
    def state(self):
        ret = np.asarray(self.last_screens)
        assert ret.shape == (4, 84, 84)
        return ret

    @property
    def is_terminal(self):
        return self.lives_lost or self.ale.game_over()

    @property
    def reward(self):
        if self._reward > 0:
            return 1
        elif self._reward < 0:
            return -1
        else:
            return 0

    @property
    def legal_actions(self):
        return self.ale.getLegalActionSet()

    def receive_action(self, action):
        assert not self.is_terminal

        self._reward = self.ale.act(action)
        self.last_screens.append(self.current_screen())
        if self.lives > self.ale.lives():
            self.lives_lost = True
        else:
            self.lives_lost = False
        self.lives = self.ale.lives()

    def initialize(self):

        if self.ale.getFrameNumber() == 0 or self.ale.game_over():
            self.ale.reset_game()
            self._reward = 0

            self.last_screens = collections.deque(
                [self.current_screen()] * self.n_last_screens,
                maxlen=self.n_last_screens)

        self.lives_lost = False
        self.lives = self.ale.lives()
