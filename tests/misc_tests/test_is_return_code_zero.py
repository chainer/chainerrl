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
