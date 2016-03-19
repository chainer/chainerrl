import unittest
import random
import tempfile
import sys

from PIL import Image
import numpy as np

import ale


class TestALE(unittest.TestCase):

    def setUp(self):
        pass

    def test_state(self):
        env = ale.ALE('breakout.bin')
        self.assertEquals(env.state.shape, (4, 84, 84))
        self.assertEquals(env.state.dtype, np.float32)
        # Pixel values must be in [-1,1]
        self.assertEquals((env.state > 1.0).sum(), 0)
        self.assertEquals((env.state < -1.0).sum(), 0)

    def test_episode(self):
        env = ale.ALE('breakout.bin')
        self.assertFalse(env.is_terminal)
        last_state = env.state
        while not env.is_terminal:
            self.assertEquals(env.state.shape, (4, 84, 84))
            print 'state', env.state.sum()
            legal_actions = env.legal_actions
            print 'legal_actions:', legal_actions
            self.assertGreater(len(legal_actions), 0)
            a = random.randrange(len(legal_actions))
            print 'a', a
            env.receive_action(a)
            np.testing.assert_array_equal(last_state[1:], env.state[:3])
            last_state = env.state

    def test_current_screen(self):
        env = ale.ALE('breakout.bin')
        tempdir = tempfile.mkdtemp()
        print >> sys.stderr, 'tempdir: {}'.format(tempdir)
        for episode in xrange(6):
            env.initialize()
            t = 0
            while not env.is_terminal:
                screen = env.state[-1]
                screen *= 128
                screen += 128
                self.assertEquals((screen > 255).sum(), 0)
                self.assertEquals((screen < 0).sum(), 0)
                legal_actions = env.legal_actions
                a = random.randrange(len(legal_actions))
                env.receive_action(a)
                img = Image.fromarray(screen.astype(np.uint8), mode='L')
                img.save('{}/{}_{}.bmp'.format(tempdir,
                                               str(episode).zfill(6), str(t).zfill(6)))
                t += 1

    def test_reward(self):
        env = ale.ALE('pong.bin')
        for episode in xrange(3):
            total_r = 0
            while not env.is_terminal:
                self.assertEquals(env.state.shape, (4, 84, 84))
                a = random.randrange(len(env.legal_actions))
                env.receive_action(a)
                total_r += env.reward
            self.assertGreater(total_r, -22)
            self.assertLess(total_r, -15)
            env.initialize()
