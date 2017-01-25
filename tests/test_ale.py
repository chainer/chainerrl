from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import random
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from chainerrl.envs import ale


class TestALE(unittest.TestCase):

    def setUp(self):
        pass

    def test_state(self):
        env = ale.ALE('breakout')
        self.assertEqual(len(env.state), 4)
        for s in env.state:
            self.assertEqual(s.shape, (84, 84))
            self.assertEqual(s.dtype, np.uint8)

    def test_episode(self):
        env = ale.ALE('breakout')
        self.assertFalse(env.is_terminal)
        last_state = env.state
        while not env.is_terminal:

            # test state
            self.assertEqual(len(env.state), 4)
            for s in env.state:
                self.assertEqual(s.shape, (84, 84))
                self.assertEqual(s.dtype, np.uint8)

            print('state (sum)', sum(env.state).sum())

            legal_actions = env.legal_actions
            print('legal_actions:', legal_actions)
            self.assertGreater(len(legal_actions), 0)
            a = random.randrange(len(legal_actions))
            print('a', a)
            env.receive_action(a)
            if not env.is_terminal:
                np.testing.assert_array_equal(
                    np.asarray(last_state[1:]), np.asarray(env.state[:3]))
            last_state = env.state

    def test_current_screen(self):
        env = ale.ALE('breakout')
        tempdir = tempfile.mkdtemp()
        print('tempdir: {}'.format(tempdir), file=sys.stderr)
        for episode in range(6):
            env.initialize()
            t = 0
            while not env.is_terminal:
                for i in range(4):
                    screen = env.state[i]
                    self.assertEqual(screen.dtype, np.uint8)
                    img = Image.fromarray(screen, mode='L')
                    filename = '{}/{}_{}_{}.bmp'.format(
                        tempdir, str(episode).zfill(6), str(t).zfill(6), i)
                    img.save(filename)
                legal_actions = env.legal_actions
                a = random.randrange(len(legal_actions))
                env.receive_action(a)
                t += 1

    def test_reward(self):
        env = ale.ALE('pong')
        for episode in range(3):
            total_r = 0
            while not env.is_terminal:
                a = random.randrange(len(env.legal_actions))
                env.receive_action(a)
                total_r += env.reward
            self.assertGreater(total_r, -22)
            self.assertLess(total_r, -15)
            env.initialize()
