from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

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


class TestAttributeSavingMixin(unittest.TestCase):

    def test_save_load(self):
        parent = Parent()
        parent.link.param.data[:] = 1
        parent.child.link.param.data[:] = 2
        # Save
        dirname = tempfile.mkdtemp()
        parent.save(dirname)
        assert os.path.isdir(dirname)
        assert os.path.isfile(os.path.join(dirname, 'link.npz'))
        assert os.path.isdir(os.path.join(dirname, 'child'))
        assert os.path.isfile(os.path.join(dirname, 'child', 'link.npz'))
        # Load
        parent = Parent()
        assert int(parent.link.param.data[:]) == 0
        assert int(parent.child.link.param.data[:]) == 0
        parent.load(dirname)
        assert int(parent.link.param.data[:]) == 1
        assert int(parent.child.link.param.data[:]) == 2
