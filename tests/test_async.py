import unittest

import chainer
import chainer.links as L
import numpy as np

import async


class TestAsync(unittest.TestCase):

    def setUp(self):
        pass

    def test_shared_link(self):
        """Check interprocess parameter sharing works if models share links
        """

        head = L.Linear(2, 2)
        model_a = chainer.ChainList(head.copy(), L.Linear(2, 3))
        model_b = chainer.ChainList(head.copy(), L.Linear(2, 4))

        a_arrays = async.extract_params_as_shared_arrays(
            chainer.ChainList(model_a))
        b_arrays = async.extract_params_as_shared_arrays(
            chainer.ChainList(model_b))

        print 'model_a shared_arrays', a_arrays
        print 'model_b shared_arrays', b_arrays

        head = L.Linear(2, 2)
        model_a = chainer.ChainList(head.copy(), L.Linear(2, 3))
        model_b = chainer.ChainList(head.copy(), L.Linear(2, 4))

        async.set_shared_params(model_a, a_arrays)
        async.set_shared_params(model_b, b_arrays)

        print 'model_a replaced'
        a_params = dict(model_a.namedparams())
        for param_name, param in a_params.iteritems():
            print param_name, param.data.ctypes.data

        print 'model_b replaced'
        b_params = dict(model_b.namedparams())
        for param_name, param in b_params.iteritems():
            print param_name, param.data.ctypes.data

        # Pointers to head parameters must be the same
        self.assertEquals(a_params['/0/W'].data.ctypes.data,
                          b_params['/0/W'].data.ctypes.data)
        self.assertEquals(a_params['/0/b'].data.ctypes.data,
                          b_params['/0/b'].data.ctypes.data)

        # Pointers to tail parameters must be different
        self.assertNotEquals(a_params['/1/W'].data.ctypes.data,
                             b_params['/1/W'].data.ctypes.data)
        self.assertNotEquals(a_params['/1/b'].data.ctypes.data,
                             b_params['/1/b'].data.ctypes.data)
