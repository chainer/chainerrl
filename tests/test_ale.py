import unittest
import random

from PIL import Image
import numpy as np

import ale


class TestALE(unittest.TestCase):

    def setUp(self):
        self.env = ale.ALE('pong.bin')

    def test_episode(self):
        self.env.initialize()
        self.assertFalse(self.env.is_terminal)
        while not self.env.is_terminal:
            # print 'state', self.env.state
            legal_actions = self.env.legal_actions
            print 'legal_actions:', legal_actions
            self.assertGreater(len(legal_actions), 0)
            a = random.randrange(len(legal_actions))
            print 'a', a
            self.env.receive_action(a)

    def test_current_screen(self):
        self.env.initialize()
        screen = self.env.current_screen()
        img = Image.fromarray(screen.astype(np.uint8), mode='L')
        img.save('test_screen.png')
