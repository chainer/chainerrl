from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()

import random
import unittest

import chainer
from chainer.testing import attr
import numpy as np

import chainerrl


class TestSetRandomSeed(unittest.TestCase):

    def test_random(self):
        chainerrl.misc.set_random_seed(0)
        seed0_0 = random.random()
        chainerrl.misc.set_random_seed(1)
        seed1_0 = random.random()
        chainerrl.misc.set_random_seed(0)
        seed0_1 = random.random()
        chainerrl.misc.set_random_seed(1)
        seed1_1 = random.random()
        self.assertEqual(seed0_0, seed0_1)
        self.assertEqual(seed1_0, seed1_1)
        self.assertNotEqual(seed0_0, seed1_0)

    def _test_xp_random(self, xp, gpus):
        chainerrl.misc.set_random_seed(0, gpus=gpus)
        seed0_0 = xp.random.rand()
        chainerrl.misc.set_random_seed(1, gpus=gpus)
        seed1_0 = xp.random.rand()
        chainerrl.misc.set_random_seed(0, gpus=gpus)
        seed0_1 = xp.random.rand()
        chainerrl.misc.set_random_seed(1, gpus=gpus)
        seed1_1 = xp.random.rand()
        self.assertEqual(seed0_0, seed0_1)
        self.assertEqual(seed1_0, seed1_1)
        self.assertNotEqual(seed0_0, seed1_0)

    def test_numpy_random(self):
        self._test_xp_random(np, gpus=())
        # It should ignore negative device IDs
        self._test_xp_random(np, gpus=(-1,))

    @attr.gpu
    def test_cupy_random(self):
        self._test_xp_random(chainer.cuda.cupy, gpus=(0,))


class TestSampleFromSpace(unittest.TestCase):

    def test_discrete(self):
        from gym import spaces
        space = spaces.Discrete(10000)
        np.random.seed(0)
        a = chainerrl.misc.sample_from_space(space)
        np.random.seed(0)
        b = chainerrl.misc.sample_from_space(space)
        self.assertTrue(space.contains(a))
        self.assertTrue(space.contains(b))
        self.assertEqual(a, b)

    def test_box(self):
        from gym import spaces
        space = spaces.Box(low=-1, high=1, shape=10)
        np.random.seed(0)
        a0 = chainerrl.misc.sample_from_space(space)
        np.random.seed(1)
        a1 = chainerrl.misc.sample_from_space(space)
        np.random.seed(0)
        b0 = chainerrl.misc.sample_from_space(space)
        np.random.seed(1)
        b1 = chainerrl.misc.sample_from_space(space)
        self.assertTrue(space.contains(a0))
        self.assertTrue(space.contains(b0))
        self.assertTrue(space.contains(a1))
        self.assertTrue(space.contains(b1))
        self.assertTrue((a0 == b0).all())
        self.assertTrue((a1 == b1).all())
        self.assertTrue((a0 != a1).all())
