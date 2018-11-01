from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import os
import tempfile
import unittest

import chainer
import numpy as np

import chainerrl


def create_simple_link():
    link = chainer.Link()
    with link.init_scope():
        link.param = chainer.Parameter(np.zeros(1))
    return link


class Parent(chainerrl.agent.AttributeSavingMixin, object):

    saved_attributes = ['link', 'child']

    def __init__(self):
        self.link = create_simple_link()
        self.child = Child()


class Child(chainerrl.agent.AttributeSavingMixin, object):

    saved_attributes = ['link']

    def __init__(self):
        self.link = create_simple_link()


class Parent2(chainerrl.agent.AttributeSavingMixin, object):

    saved_attributes = ['child_a', 'child_b']

    def __init__(self, child_a, child_b):
        self.child_a = child_a
        self.child_b = child_b


class TestAttributeSavingMixin(unittest.TestCase):

    def test_save_load(self):
        parent = Parent()
        parent.link.param.array[:] = 1
        parent.child.link.param.array[:] = 2
        # Save
        dirname = tempfile.mkdtemp()
        parent.save(dirname)
        self.assertTrue(os.path.isdir(dirname))
        self.assertTrue(os.path.isfile(os.path.join(dirname, 'link.npz')))
        self.assertTrue(os.path.isdir(os.path.join(dirname, 'child')))
        self.assertTrue(os.path.isfile(
            os.path.join(dirname, 'child', 'link.npz')))
        # Load
        parent = Parent()
        self.assertEqual(int(parent.link.param.array), 0)
        self.assertEqual(int(parent.child.link.param.array), 0)
        parent.load(dirname)
        self.assertEqual(int(parent.link.param.array), 1)
        self.assertEqual(int(parent.child.link.param.array), 2)

    def test_save_load_2(self):
        parent = Parent()
        parent2 = Parent2(parent.child, parent)
        # Save
        dirname = tempfile.mkdtemp()
        parent2.save(dirname)
        # Load
        parent = Parent()
        parent2 = Parent2(parent.child, parent)
        parent2.load(dirname)

    def test_loop1(self):
        parent = Parent()
        parent.child = parent
        dirname = tempfile.mkdtemp()

        # The assertion in ChainerRL should fail on save().
        # Otherwise it seems to raise OSError: [Errno 63] File name too long
        with self.assertRaises(AssertionError):
            parent.save(dirname)

    def test_loop2(self):
        parent1 = Parent()
        parent2 = Parent()
        parent1.child = parent2
        parent2.child = parent1
        dirname = tempfile.mkdtemp()

        # The assertion in ChainerRL should fail on save().
        # Otherwise it seems to raise OSError: [Errno 63] File name too long
        with self.assertRaises(AssertionError):
            parent1.save(dirname)
