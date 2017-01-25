from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import multiprocessing as mp
import unittest

import chainer
import chainer.links as L
from chainer import optimizers
import numpy as np

from chainerrl.misc import async


class TestAsync(unittest.TestCase):

    def setUp(self):
        pass

    def test_share_params(self):

        # A's params are shared with B and C so that all the three share the
        # same parameter arrays

        model_a = L.Linear(2, 2)

        arrays = async.share_params_as_shared_arrays(model_a)

        model_b = L.Linear(2, 2)
        model_c = L.Linear(2, 2)

        async.set_shared_params(model_b, arrays)
        async.set_shared_params(model_c, arrays)

        a_params = dict(model_a.namedparams())
        b_params = dict(model_b.namedparams())
        c_params = dict(model_c.namedparams())

        def assert_same_pointers_to_data(a, b):
            self.assertEqual(a['/W'].data.ctypes.data,
                             b['/W'].data.ctypes.data)
            self.assertEqual(a['/b'].data.ctypes.data,
                             b['/b'].data.ctypes.data)

        def assert_different_pointers_to_grad(a, b):
            self.assertNotEqual(a['/W'].grad.ctypes.data,
                                b['/W'].grad.ctypes.data)
            self.assertNotEqual(a['/b'].grad.ctypes.data,
                                b['/b'].grad.ctypes.data)

        # Pointers to parameters must be the same
        assert_same_pointers_to_data(a_params, b_params)
        assert_same_pointers_to_data(a_params, c_params)
        # Pointers to gradients must be different
        assert_different_pointers_to_grad(a_params, b_params)
        assert_different_pointers_to_grad(a_params, c_params)

    def test_share_states(self):

        model = L.Linear(2, 2)
        opt_a = optimizers.RMSprop()
        opt_a.setup(model)
        arrays = async.share_states_as_shared_arrays(opt_a)
        opt_b = optimizers.RMSprop()
        opt_b.setup(model)
        opt_c = optimizers.RMSprop()
        opt_c.setup(model)

        def assert_different_pointers(a, b):
            self.assertTrue(a)
            for param_name in a:
                self.assertTrue(a[param_name])
                for state_name in a[param_name]:
                    self.assertTrue(isinstance(
                        a[param_name][state_name], np.ndarray))
                    self.assertTrue(isinstance(
                        b[param_name][state_name], np.ndarray))
                    self.assertNotEqual(a[param_name][state_name].ctypes.data,
                                        b[param_name][state_name].ctypes.data)

        assert_different_pointers(opt_a._states, opt_b._states)
        assert_different_pointers(opt_a._states, opt_c._states)

        async.set_shared_states(opt_b, arrays)
        async.set_shared_states(opt_c, arrays)

        def assert_same_pointers(a, b):
            self.assertTrue(a)
            for param_name in a:
                self.assertTrue(a[param_name])
                for state_name in a[param_name]:
                    self.assertTrue(isinstance(
                        a[param_name][state_name], np.ndarray))
                    self.assertTrue(isinstance(
                        b[param_name][state_name], np.ndarray))
                    self.assertEqual(a[param_name][state_name].ctypes.data,
                                     b[param_name][state_name].ctypes.data)

        assert_same_pointers(opt_a._states, opt_b._states)
        assert_same_pointers(opt_a._states, opt_c._states)

    def test_shared_link(self):
        """Check interprocess parameter sharing works if models share links"""

        head = L.Linear(2, 2)
        model_a = chainer.ChainList(head.copy(), L.Linear(2, 3))
        model_b = chainer.ChainList(head.copy(), L.Linear(2, 4))

        a_arrays = async.extract_params_as_shared_arrays(
            chainer.ChainList(model_a))
        b_arrays = async.extract_params_as_shared_arrays(
            chainer.ChainList(model_b))

        print(('model_a shared_arrays', a_arrays))
        print(('model_b shared_arrays', b_arrays))

        head = L.Linear(2, 2)
        model_a = chainer.ChainList(head.copy(), L.Linear(2, 3))
        model_b = chainer.ChainList(head.copy(), L.Linear(2, 4))

        async.set_shared_params(model_a, a_arrays)
        async.set_shared_params(model_b, b_arrays)

        print('model_a replaced')
        a_params = dict(model_a.namedparams())
        for param_name, param in list(a_params.items()):
            print((param_name, param.data.ctypes.data))

        print('model_b replaced')
        b_params = dict(model_b.namedparams())
        for param_name, param in list(b_params.items()):
            print((param_name, param.data.ctypes.data))

        # Pointers to head parameters must be the same
        self.assertEqual(a_params['/0/W'].data.ctypes.data,
                         b_params['/0/W'].data.ctypes.data)
        self.assertEqual(a_params['/0/b'].data.ctypes.data,
                         b_params['/0/b'].data.ctypes.data)

        # Pointers to tail parameters must be different
        self.assertNotEqual(a_params['/1/W'].data.ctypes.data,
                            b_params['/1/W'].data.ctypes.data)
        self.assertNotEqual(a_params['/1/b'].data.ctypes.data,
                            b_params['/1/b'].data.ctypes.data)

    def test_shared_link_copy(self):
        head = L.Linear(2, 2)
        model_a = chainer.ChainList(head.copy(), L.Linear(2, 3))
        model_b = chainer.ChainList(head.copy(), L.Linear(2, 4))
        a_params = dict(model_a.namedparams())
        b_params = dict(model_b.namedparams())
        self.assertEqual(a_params['/0/W'].data.ctypes.data,
                         b_params['/0/W'].data.ctypes.data)
        self.assertEqual(a_params['/0/b'].data.ctypes.data,
                         b_params['/0/b'].data.ctypes.data)
        import copy
        model_a_copy = copy.deepcopy(model_a)
        model_b_copy = copy.deepcopy(model_b)
        a_copy_params = dict(model_a_copy.namedparams())
        b_copy_params = dict(model_b_copy.namedparams())
        # When A and B are separately deepcopied, head is no longer shared
        self.assertNotEqual(a_copy_params['/0/W'].data.ctypes.data,
                            b_copy_params['/0/W'].data.ctypes.data)
        self.assertNotEqual(a_copy_params['/0/b'].data.ctypes.data,
                            b_copy_params['/0/b'].data.ctypes.data)

        model_total_copy = copy.deepcopy(chainer.ChainList(model_a, model_b))
        model_a_copy = model_total_copy[0]
        model_b_copy = model_total_copy[1]
        a_copy_params = dict(model_a_copy.namedparams())
        b_copy_params = dict(model_b_copy.namedparams())
        # When ChainList(A, B) is deepcopied, head is still shared!
        self.assertEqual(a_copy_params['/0/W'].data.ctypes.data,
                         b_copy_params['/0/W'].data.ctypes.data)
        self.assertEqual(a_copy_params['/0/b'].data.ctypes.data,
                         b_copy_params['/0/b'].data.ctypes.data)

    def test_run_async(self):
        counter = mp.Value('l', 0)

        def run_func(process_idx):
            for _ in range(1000):
                with counter.get_lock():
                    counter.value += 1
        async.run_async(4, run_func)
        self.assertEqual(counter.value, 4000)
