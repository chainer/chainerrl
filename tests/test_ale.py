import unittest
import random
import tempfile
import sys

from PIL import Image
import numpy as np

import ale


class TestALE(unittest.TestCase):

    def setUp(self):
        self.env = ale.ALE('pong.bin')

    def test_state(self):
        self.env.initialize()
        self.assertEquals(self.env.state.shape, (4, 84, 84))
        self.assertEquals(self.env.state.dtype, np.float32)
        # Pixel values must be in [-1,1]
        self.assertEquals((self.env.state > 1.0).sum(), 0)
        self.assertEquals((self.env.state < -1.0).sum(), 0)

    def test_episode(self):
        self.env.initialize()
        self.assertFalse(self.env.is_terminal)
        last_state = self.env.state
        while not self.env.is_terminal:
            self.assertEquals(self.env.state.shape, (4, 84, 84))
            print 'state', self.env.state.sum()
            legal_actions = self.env.legal_actions
            print 'legal_actions:', legal_actions
            self.assertGreater(len(legal_actions), 0)
            a = random.randrange(len(legal_actions))
            print 'a', a
            self.env.receive_action(a)
            np.testing.assert_array_equal(last_state[1:], self.env.state[:3])
            last_state = self.env.state

    def test_current_screen(self):
        self.env.initialize()
        tempdir = tempfile.mkdtemp()
        print >> sys.stderr, 'tempdir: {}'.format(tempdir)
        t = 0
        while not self.env.is_terminal:
            screen = self.env.current_screen()
            screen *= 128
            screen += 128
            self.assertEquals((screen > 255).sum(), 0)
            self.assertEquals((screen < 0).sum(), 0)
            legal_actions = self.env.legal_actions
            a = random.randrange(len(legal_actions))
            self.env.receive_action(a)
            img = Image.fromarray(screen.astype(np.uint8), mode='L')
            img.save('{}/{}.bmp'.format(tempdir, str(t).zfill(6)))
            t += 1
