from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import multiprocessing as mp
import os
import signal
import sys
import unittest
import warnings

import chainer
import chainer.links as L
from chainer import optimizers
import copy
import numpy as np

from chainerrl.misc import async_


class TestAsync(unittest.TestCase):

    def setUp(self):
        pass

    def test_share_params(self):

        # A's params are shared with B and C so that all the three share the
        # same parameter arrays

        model_a = L.Linear(2, 2)

        arrays = async_.share_params_as_shared_arrays(model_a)

        model_b = L.Linear(2, 2)
        model_c = L.Linear(2, 2)

        async_.set_shared_params(model_b, arrays)
        async_.set_shared_params(model_c, arrays)

        a_params = dict(model_a.namedparams())
        b_params = dict(model_b.namedparams())
        c_params = dict(model_c.namedparams())

        def assert_same_pointers_to_data(a, b):
            self.assertEqual(a['/W'].array.ctypes.data,
                             b['/W'].array.ctypes.data)
            self.assertEqual(a['/b'].array.ctypes.data,
                             b['/b'].array.ctypes.data)

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
        arrays = async_.share_states_as_shared_arrays(opt_a)
        opt_b = optimizers.RMSprop()
        opt_b.setup(copy.deepcopy(model))
        # In Chainer v2, a model cannot be set up by two optimizers or more.

        opt_c = optimizers.RMSprop()
        opt_c.setup(copy.deepcopy(model))

        """
        Removed the tests by assert_different_pointers
        since they are trivial now.
        """

        async_.set_shared_states(opt_b, arrays)
        async_.set_shared_states(opt_c, arrays)

        def assert_same_pointers(a, b):
            a = a.target
            b = b.target
            for param_name, param_a in a.namedparams():
                param_b = dict(b.namedparams())[param_name]
                state_a = param_a.update_rule.state
                state_b = param_b.update_rule.state
                self.assertTrue(state_a)
                self.assertTrue(state_b)
                for state_name, state_val_a in state_a.items():
                    state_val_b = state_b[state_name]
                    self.assertTrue(isinstance(
                        state_val_a, np.ndarray))
                    self.assertTrue(isinstance(
                        state_val_b, np.ndarray))
                    self.assertEqual(state_val_a.ctypes.data,
                                     state_val_b.ctypes.data)

        assert_same_pointers(opt_a, opt_b)
        assert_same_pointers(opt_a, opt_c)

    def test_shared_link(self):
        """Check interprocess parameter sharing works if models share links"""

        head = L.Linear(2, 2)
        model_a = chainer.ChainList(head.copy(), L.Linear(2, 3))
        model_b = chainer.ChainList(head.copy(), L.Linear(2, 4))

        a_arrays = async_.extract_params_as_shared_arrays(
            chainer.ChainList(model_a))
        b_arrays = async_.extract_params_as_shared_arrays(
            chainer.ChainList(model_b))

        print(('model_a shared_arrays', a_arrays))
        print(('model_b shared_arrays', b_arrays))

        head = L.Linear(2, 2)
        model_a = chainer.ChainList(head.copy(), L.Linear(2, 3))
        model_b = chainer.ChainList(head.copy(), L.Linear(2, 4))

        async_.set_shared_params(model_a, a_arrays)
        async_.set_shared_params(model_b, b_arrays)

        print('model_a replaced')
        a_params = dict(model_a.namedparams())
        for param_name, param in list(a_params.items()):
            print((param_name, param.array.ctypes.data))

        print('model_b replaced')
        b_params = dict(model_b.namedparams())
        for param_name, param in list(b_params.items()):
            print((param_name, param.array.ctypes.data))

        # Pointers to head parameters must be the same
        self.assertEqual(a_params['/0/W'].array.ctypes.data,
                         b_params['/0/W'].array.ctypes.data)
        self.assertEqual(a_params['/0/b'].array.ctypes.data,
                         b_params['/0/b'].array.ctypes.data)

        # Pointers to tail parameters must be different
        self.assertNotEqual(a_params['/1/W'].array.ctypes.data,
                            b_params['/1/W'].array.ctypes.data)
        self.assertNotEqual(a_params['/1/b'].array.ctypes.data,
                            b_params['/1/b'].array.ctypes.data)

    def test_shared_link_copy(self):
        head = L.Linear(2, 2)
        model_a = chainer.ChainList(head.copy(), L.Linear(2, 3))
        model_b = chainer.ChainList(head.copy(), L.Linear(2, 4))
        a_params = dict(model_a.namedparams())
        b_params = dict(model_b.namedparams())
        self.assertEqual(a_params['/0/W'].array.ctypes.data,
                         b_params['/0/W'].array.ctypes.data)
        self.assertEqual(a_params['/0/b'].array.ctypes.data,
                         b_params['/0/b'].array.ctypes.data)
        import copy
        model_a_copy = copy.deepcopy(model_a)
        model_b_copy = copy.deepcopy(model_b)
        a_copy_params = dict(model_a_copy.namedparams())
        b_copy_params = dict(model_b_copy.namedparams())
        # When A and B are separately deepcopied, head is no longer shared
        self.assertNotEqual(a_copy_params['/0/W'].array.ctypes.data,
                            b_copy_params['/0/W'].array.ctypes.data)
        self.assertNotEqual(a_copy_params['/0/b'].array.ctypes.data,
                            b_copy_params['/0/b'].array.ctypes.data)

        model_total_copy = copy.deepcopy(chainer.ChainList(model_a, model_b))
        model_a_copy = model_total_copy[0]
        model_b_copy = model_total_copy[1]
        a_copy_params = dict(model_a_copy.namedparams())
        b_copy_params = dict(model_b_copy.namedparams())
        # When ChainList(A, B) is deepcopied, head is still shared!
        self.assertEqual(a_copy_params['/0/W'].array.ctypes.data,
                         b_copy_params['/0/W'].array.ctypes.data)
        self.assertEqual(a_copy_params['/0/b'].array.ctypes.data,
                         b_copy_params['/0/b'].array.ctypes.data)

    def test_run_async(self):
        counter = mp.Value('l', 0)

        def run_func(process_idx):
            for _ in range(1000):
                with counter.get_lock():
                    counter.value += 1
        async_.run_async(4, run_func)
        self.assertEqual(counter.value, 4000)

    def test_run_async_exit_code(self):

        def run_with_exit_code_0(process_idx):
            sys.exit(0)

        def run_with_exit_code_11(process_idx):
            os.kill(os.getpid(), signal.SIGSEGV)

        with warnings.catch_warnings(record=True) as ws:
            async_.run_async(4, run_with_exit_code_0)
            # There should be no AbnormalExitWarning
            self.assertEqual(
                sum(1 if issubclass(
                    w.category, async_.AbnormalExitWarning) else 0
                    for w in ws), 0)

        with warnings.catch_warnings(record=True) as ws:
            async_.run_async(4, run_with_exit_code_11)
            # There should be 4 AbnormalExitWarning
            self.assertEqual(
                sum(1 if issubclass(
                    w.category, async_.AbnormalExitWarning) else 0
                    for w in ws), 4)
