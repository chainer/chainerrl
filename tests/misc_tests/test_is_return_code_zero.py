from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

import chainerrl


class TestIsReturnCodeZero(unittest.TestCase):

    def test(self):
        # Assume ls command exists
        self.assertTrue(chainerrl.misc.is_return_code_zero(['ls']))
        self.assertFalse(chainerrl.misc.is_return_code_zero(
            ['ls --nonexistentoption']))
        self.assertFalse(chainerrl.misc.is_return_code_zero(
            ['nonexistentcommand']))
