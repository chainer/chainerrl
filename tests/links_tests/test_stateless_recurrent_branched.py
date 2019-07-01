from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import testing
import numpy as np

from chainerrl.links import StatelessRecurrentBranched
from chainerrl.links import StatelessRecurrentSequential


class TestStatelessRecurrentBranched(unittest.TestCase):

    def _test_n_step_forward(self, gpu):
        in_size = 2
        out0_size = 3
        out1_size = 4
        out2_size = 1

        par = StatelessRecurrentBranched(
            L.NStepLSTM(1, in_size, out0_size, 0),
            StatelessRecurrentSequential(
                L.NStepRNNReLU(1, in_size, out1_size, 0),
            ),
            StatelessRecurrentSequential(
                L.Linear(in_size, out2_size),
            ),
        )

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            par.to_gpu()
        xp = par.xp

        seqs_x = [
            xp.random.uniform(-1, 1, size=(1, in_size)).astype(np.float32),
            xp.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
        ]

        # Concatenated output should be a tuple of three variables.
        concat_out, concat_rs = par.n_step_forward(
            seqs_x, None, output_mode='concat')
        self.assertIsInstance(concat_out, tuple)
        self.assertEqual(len(concat_out), len(par))
        self.assertEqual(concat_out[0].shape, (4, out0_size))
        self.assertEqual(concat_out[1].shape, (4, out1_size))
        self.assertEqual(concat_out[2].shape, (4, out2_size))

        self.assertIsInstance(concat_rs, tuple)
        self.assertEqual(len(concat_rs), len(par))
        self.assertIsInstance(concat_rs[0], tuple)
        # NStepLSTM
        self.assertEqual(len(concat_rs[0]), 2)
        self.assertEqual(concat_rs[0][0].shape, (1, len(seqs_x), out0_size))
        self.assertEqual(concat_rs[0][1].shape, (1, len(seqs_x), out0_size))
        # StatelessRecurrentSequential(NStepRNNReLU)
        self.assertEqual(len(concat_rs[1]), 1)
        self.assertEqual(concat_rs[1][0].shape, (1, len(seqs_x), out1_size))
        # StatelessRecurrentSequential(Linear)
        self.assertEqual(len(concat_rs[2]), 0)

        # Split output should be a list of two tuples, each of which is of
        # three variables.
        split_out, split_rs = par.n_step_forward(
            seqs_x, None, output_mode='split')
        self.assertIsInstance(split_out, list)
        self.assertEqual(len(split_out), len(seqs_x))
        self.assertEqual(len(split_out[0]), len(par))
        self.assertEqual(len(split_out[1]), len(par))
        self.assertEqual(split_out[0][0].shape, (1, out0_size,))
        self.assertEqual(split_out[0][1].shape, (1, out1_size,))
        self.assertEqual(split_out[0][2].shape, (1, out2_size,))
        self.assertEqual(split_out[1][0].shape, (3, out0_size,))
        self.assertEqual(split_out[1][1].shape, (3, out1_size,))
        self.assertEqual(split_out[1][2].shape, (3, out2_size,))

        # Check if output_mode='concat' and output_mode='split' are consistent
        xp.testing.assert_allclose(
            F.concat([F.concat(seq_out, axis=1)
                      for seq_out in split_out], axis=0).array,
            F.concat(concat_out, axis=1).array,
        )

    @testing.attr.gpu
    def test_n_step_forward_gpu(self):
        self._test_n_step_forward(gpu=0)

    def test_n_step_forward_cpu(self):
        self._test_n_step_forward(gpu=-1)

    def _test_mask_recurrent_state_at(self, gpu):
        in_size = 2
        out0_size = 2
        out1_size = 3
        par = StatelessRecurrentBranched(
            L.NStepGRU(1, in_size, out0_size, 0),
            StatelessRecurrentSequential(
                L.NStepLSTM(1, in_size, out1_size, 0),
            ),
        )
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            par.to_gpu()
        xp = par.xp
        seqs_x = [
            xp.random.uniform(-1, 1, size=(2, in_size)).astype(np.float32),
            xp.random.uniform(-1, 1, size=(2, in_size)).astype(np.float32),
        ]
        transposed_x = F.transpose_sequence(seqs_x)

        nstep_out, nstep_rs = par.n_step_forward(
            seqs_x, None, output_mode='concat')

        # Check if n_step_forward and forward twice results are same
        def no_mask_forward_twice():
            _, rs = par(transposed_x[0], None)
            return par(transposed_x[1], rs)

        nomask_out, nomask_rs = no_mask_forward_twice()
        # GRU
        xp.testing.assert_allclose(
            nstep_out[0].array[[1, 3]],
            nomask_out[0].array,
        )
        # LSTM
        xp.testing.assert_allclose(
            nstep_out[1].array[[1, 3]],
            nomask_out[1].array,
        )
        xp.testing.assert_allclose(nstep_rs[0].array, nomask_rs[0].array)
        self.assertIsInstance(nomask_rs[1], tuple)
        self.assertEqual(len(nomask_rs[1]), 1)
        self.assertEqual(len(nomask_rs[1][0]), 2)
        xp.testing.assert_allclose(
            nstep_rs[1][0][0].array, nomask_rs[1][0][0].array)
        xp.testing.assert_allclose(
            nstep_rs[1][0][1].array, nomask_rs[1][0][1].array)

        # 1st-only mask forward twice: only 2nd should be the same
        def mask0_forward_twice():
            _, rs = par(transposed_x[0], None)
            rs = par.mask_recurrent_state_at(rs, 0)
            return par(transposed_x[1], rs)
        mask0_out, mask0_rs = mask0_forward_twice()
        # GRU
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out[0].array[1],
                mask0_out[0].array[0],
            )
        xp.testing.assert_allclose(
            nstep_out[0].array[3],
            mask0_out[0].array[1],
        )
        # LSTM
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out[1].array[1],
                mask0_out[1].array[0],
            )
        xp.testing.assert_allclose(
            nstep_out[1].array[3],
            mask0_out[1].array[1],
        )

        # 2nd-only mask forward twice: only 1st should be the same
        def mask1_forward_twice():
            _, rs = par(transposed_x[0], None)
            rs = par.mask_recurrent_state_at(rs, 1)
            return par(transposed_x[1], rs)
        mask1_out, mask1_rs = mask1_forward_twice()
        # GRU
        xp.testing.assert_allclose(
            nstep_out[0].array[1],
            mask1_out[0].array[0],
        )
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out[0].array[3],
                mask1_out[0].array[1],
            )
        # LSTM
        xp.testing.assert_allclose(
            nstep_out[1].array[1],
            mask1_out[1].array[0],
        )
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out[1].array[3],
                mask1_out[1].array[1],
            )

        # both 1st and 2nd mask forward twice: both should be different
        def mask01_forward_twice():
            _, rs = par(transposed_x[0], None)
            rs = par.mask_recurrent_state_at(rs, [0, 1])
            return par(transposed_x[1], rs)
        mask01_out, mask01_rs = mask01_forward_twice()
        # GRU
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out[0].array[1],
                mask01_out[0].array[0],
            )
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out[0].array[3],
                mask01_out[0].array[1],
            )
        # LSTM
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out[1].array[1],
                mask01_out[1].array[0],
            )
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out[1].array[3],
                mask01_out[1].array[1],
            )

        # get and concat recurrent states and resume forward
        def get_and_concat_rs_forward():
            _, rs = par(transposed_x[0], None)
            rs0 = par.get_recurrent_state_at(rs, 0, unwrap_variable=True)
            rs1 = par.get_recurrent_state_at(rs, 1, unwrap_variable=True)
            concat_rs = par.concatenate_recurrent_states([rs0, rs1])
            return par(transposed_x[1], concat_rs)
        getcon_out, getcon_rs = get_and_concat_rs_forward()
        # GRU
        xp.testing.assert_allclose(
            nstep_out[0].array[1],
            getcon_out[0].array[0],
        )
        xp.testing.assert_allclose(
            nstep_out[0].array[3],
            getcon_out[0].array[1],
        )
        # LSTM
        xp.testing.assert_allclose(
            nstep_out[1].array[1],
            getcon_out[1].array[0],
        )
        xp.testing.assert_allclose(
            nstep_out[1].array[3],
            getcon_out[1].array[1],
        )

    @testing.attr.gpu
    def test_mask_recurrent_state_at_gpu(self):
        self._test_mask_recurrent_state_at(gpu=0)

    def test_mask_recurrent_state_at_cpu(self):
        self._test_mask_recurrent_state_at(gpu=-1)
