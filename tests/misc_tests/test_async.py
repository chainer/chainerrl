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

import chainerrl
from chainerrl.misc import async_


def _assert_same_pointers_to_persistent_values(a, b):
    assert isinstance(a, chainer.Link)
    assert isinstance(b, chainer.Link)
    a_persistents = dict(chainerrl.misc.namedpersistent(a))
    b_persistents = dict(chainerrl.misc.namedpersistent(b))
    assert set(a_persistents.keys()) == set(b_persistents.keys())
    for key in a_persistents:
        a_persistent = a_persistents[key]
        b_persistent = b_persistents[key]
        assert isinstance(a_persistent, np.ndarray)
        assert isinstance(b_persistent, np.ndarray)
        assert a_persistent.ctypes.data == b_persistent.ctypes.data


def _assert_same_pointers_to_param_data(a, b):
    assert isinstance(a, chainer.Link)
    assert isinstance(b, chainer.Link)
    a_params = dict(a.namedparams())
    b_params = dict(b.namedparams())
    assert set(a_params.keys()) == set(b_params.keys())
    for key in a_params.keys():
        assert isinstance(a_params[key], chainer.Variable)
        assert isinstance(b_params[key], chainer.Variable)
        assert (a_params[key].array.ctypes.data
                == b_params[key].array.ctypes.data)


def _assert_different_pointers_to_param_grad(a, b):
    assert isinstance(a, chainer.Link)
    assert isinstance(b, chainer.Link)
    a_params = dict(a.namedparams())
    b_params = dict(b.namedparams())
    assert set(a_params.keys()) == set(b_params.keys())
    for key in a_params.keys():
        assert isinstance(a_params[key], chainer.Variable)
        assert isinstance(b_params[key], chainer.Variable)
        assert (a_params[key].grad.ctypes.data
                != b_params[key].grad.ctypes.data)


class TestAsync(unittest.TestCase):

    def setUp(self):
        pass

    def test_share_params_linear(self):

        # A's params are shared with B and C so that all the three share the
        # same parameter arrays

        model_a = L.Linear(2, 2)

        arrays = async_.share_params_as_shared_arrays(model_a)
        assert isinstance(arrays, dict)
        assert set(arrays.keys()) == {'/W', '/b'}

        model_b = L.Linear(2, 2)
        model_c = L.Linear(2, 2)

        async_.set_shared_params(model_b, arrays)
        async_.set_shared_params(model_c, arrays)

        # Pointers to parameters must be the same
        _assert_same_pointers_to_param_data(model_a, model_b)
        _assert_same_pointers_to_param_data(model_a, model_c)
        # Pointers to gradients must be different
        _assert_different_pointers_to_param_grad(model_a, model_b)
        _assert_different_pointers_to_param_grad(model_a, model_c)
        _assert_different_pointers_to_param_grad(model_b, model_c)
        # Pointers to persistent values must be the same
        _assert_same_pointers_to_persistent_values(model_a, model_b)
        _assert_same_pointers_to_persistent_values(model_a, model_c)

    def test_share_params_batch_normalization(self):

        # A's params and persistent values are all shared with B and C

        model_a = L.BatchNormalization(3)

        arrays = async_.share_params_as_shared_arrays(model_a)
        assert isinstance(arrays, dict)
        assert set(arrays.keys()) == {
            '/gamma', '/beta', '/avg_mean', '/avg_var', '/N'}

        model_b = L.BatchNormalization(3)
        model_c = L.BatchNormalization(3)

        async_.set_shared_params(model_b, arrays)
        async_.set_shared_params(model_c, arrays)

        # Pointers to parameters must be the same
        _assert_same_pointers_to_param_data(model_a, model_b)
        _assert_same_pointers_to_param_data(model_a, model_c)
        # Pointers to gradients must be different
        _assert_different_pointers_to_param_grad(model_a, model_b)
        _assert_different_pointers_to_param_grad(model_a, model_c)
        _assert_different_pointers_to_param_grad(model_b, model_c)
        # Pointers to persistent values must be the same
        _assert_same_pointers_to_persistent_values(model_a, model_b)
        _assert_same_pointers_to_persistent_values(model_a, model_c)

        # Check if N is shared correctly among links
        assert model_a.N == 0
        assert model_b.N == 0
        assert model_c.N == 0
        test_input = np.random.normal(size=(2, 3)).astype(np.float32)
        model_a(test_input, finetune=True)
        assert model_a.N == 1
        assert model_b.N == 1
        assert model_c.N == 1
        model_c(test_input, finetune=True)
        assert model_a.N == 2
        assert model_b.N == 2
        assert model_c.N == 2

    def test_share_params_chain_list(self):

        model_a = chainer.ChainList(
            L.BatchNormalization(3),
            chainer.ChainList(L.Linear(3, 5)),
        )

        arrays = async_.share_params_as_shared_arrays(model_a)
        assert isinstance(arrays, dict)
        assert set(arrays.keys()) == {
            '/0/gamma', '/0/beta', '/0/avg_mean', '/0/avg_var', '/0/N',
            '/1/0/W', '/1/0/b'}

        model_b = chainer.ChainList(
            L.BatchNormalization(3),
            chainer.ChainList(L.Linear(3, 5)),
        )
        model_c = chainer.ChainList(
            L.BatchNormalization(3),
            chainer.ChainList(L.Linear(3, 5)),
        )

        async_.set_shared_params(model_b, arrays)
        async_.set_shared_params(model_c, arrays)

        # Pointers to parameters must be the same
        _assert_same_pointers_to_param_data(model_a, model_b)
        _assert_same_pointers_to_param_data(model_a, model_c)
        # Pointers to gradients must be different
        _assert_different_pointers_to_param_grad(model_a, model_b)
        _assert_different_pointers_to_param_grad(model_a, model_c)
        _assert_different_pointers_to_param_grad(model_b, model_c)
        # Pointers to persistent values must be the same
        _assert_same_pointers_to_persistent_values(model_a, model_b)
        _assert_same_pointers_to_persistent_values(model_a, model_c)

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

        a_arrays = async_.extract_params_as_shared_arrays(model_a)
        b_arrays = async_.extract_params_as_shared_arrays(model_b)

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
