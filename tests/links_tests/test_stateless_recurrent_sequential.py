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
from chainer.functions.connection.n_step_lstm import _lstm
from chainer import links as L
from chainer import testing
import numpy as np

from chainerrl.links import StatelessRecurrentSequential


def _step_lstm(lstm, x, state):
    assert isinstance(lstm, L.NStepLSTM)
    assert len(lstm.ws) == 1
    assert len(lstm.bs) == 1
    assert len(lstm.ws[0]) == 8
    assert len(lstm.bs[0]) == 8
    if state is None or state[0] is None:
        xp = lstm.xp
        h = xp.zeros((len(x), lstm.out_size), dtype=np.float32)
        c = xp.zeros((len(x), lstm.out_size), dtype=np.float32)
    else:
        h, c = state
    h, c = _lstm(x, h, c, lstm.ws[0], lstm.bs[0])
    return h, (h, c)


def _step_rnn_tanh(rnn, x, state):
    assert isinstance(rnn, L.NStepRNNTanh)
    assert len(rnn.ws) == 1
    assert len(rnn.bs) == 1
    assert len(rnn.ws[0]) == 2
    assert len(rnn.bs[0]) == 2
    if state is None:
        xp = rnn.xp
        h = xp.zeros((len(x), rnn.out_size), dtype=np.float32)
    else:
        h = state
    w0, w1 = rnn.ws[0]
    b0, b1 = rnn.bs[0]
    h = F.tanh(F.linear(x, w0, b0) + F.linear(h, w1, b1))
    return h, h


class TestStatelessRecurrentSequential(unittest.TestCase):

    def _test_n_step_forward(self, gpu):
        in_size = 2
        out_size = 6

        rseq = StatelessRecurrentSequential(
            L.Linear(in_size, 3),
            F.elu,
            L.NStepLSTM(1, 3, 4, 0),
            L.Linear(4, 5),
            L.NStepRNNTanh(1, 5, out_size, 0),
            F.tanh,
        )

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            rseq.to_gpu()
        xp = rseq.xp

        linear1 = rseq._layers[0]
        lstm = rseq._layers[2]
        linear2 = rseq._layers[3]
        rnn = rseq._layers[4]

        seqs_x = [
            xp.random.uniform(-1, 1, size=(4, in_size)).astype(np.float32),
            xp.random.uniform(-1, 1, size=(1, in_size)).astype(np.float32),
            xp.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
        ]

        concat_out, concat_state = rseq.n_step_forward(
            seqs_x, None, output_mode='concat')
        self.assertEqual(concat_out.shape, (8, out_size))

        split_out, split_state = rseq.n_step_forward(
            seqs_x, None, output_mode='split')
        self.assertIsInstance(split_out, list)
        self.assertEqual(len(split_out), len(seqs_x))
        for seq_x, seq_out in zip(seqs_x, split_out):
            self.assertEqual(seq_out.shape, (len(seq_x), out_size))

        # Check if output_mode='concat' and output_mode='split' are consistent
        xp.testing.assert_allclose(
            F.concat(split_out, axis=0).array,
            concat_out.array,
        )

        (concat_lstm_h, concat_lstm_c), concat_rnn_h = concat_state
        (split_lstm_h, split_lstm_c), split_rnn_h = split_state
        xp.testing.assert_allclose(concat_lstm_h.array, split_lstm_h.array)
        xp.testing.assert_allclose(concat_lstm_c.array, split_lstm_c.array)
        xp.testing.assert_allclose(concat_rnn_h.array, split_rnn_h.array)

        # Check if the output matches that of step-by-step execution
        def manual_n_step_forward(seqs_x):
            sorted_seqs_x = sorted(seqs_x, key=len, reverse=True)
            transposed_x = F.transpose_sequence(sorted_seqs_x)
            lstm_h = None
            lstm_c = None
            rnn_h = None
            ys = []
            for batch in transposed_x:
                if lstm_h is not None:
                    lstm_h = lstm_h[:len(batch)]
                    lstm_c = lstm_c[:len(batch)]
                    rnn_h = rnn_h[:len(batch)]
                h = linear1(batch)
                h = F.elu(h)
                h, (lstm_h, lstm_c) = _step_lstm(lstm, h, (lstm_h, lstm_c))
                h = linear2(h)
                h, rnn_h = _step_rnn_tanh(rnn, h, rnn_h)
                y = F.tanh(h)
                ys.append(y)
            sorted_seqs_y = F.transpose_sequence(ys)
            # Undo sort
            seqs_y = [sorted_seqs_y[0], sorted_seqs_y[2], sorted_seqs_y[1]]
            return seqs_y

        manual_split_out = manual_n_step_forward(seqs_x)
        for man_seq_out, seq_out in zip(manual_split_out, split_out):
            xp.testing.assert_allclose(
                man_seq_out.array, seq_out.array, rtol=1e-5)

        # Finally, check the gradient (wrt linear1.W)
        concat_grad, = chainer.grad([F.sum(concat_out)], [linear1.W])
        split_grad, = chainer.grad(
            [F.sum(F.concat(split_out, axis=0))], [linear1.W])
        manual_split_grad, = chainer.grad(
            [F.sum(F.concat(manual_split_out, axis=0))], [linear1.W])
        xp.testing.assert_allclose(
            concat_grad.array, split_grad.array, rtol=1e-5)
        xp.testing.assert_allclose(
            concat_grad.array, manual_split_grad.array, rtol=1e-5)

    @testing.attr.gpu
    def test_n_step_forward_gpu(self):
        self._test_n_step_forward(gpu=0)

    def test_n_step_forward_cpu(self):
        self._test_n_step_forward(gpu=-1)

    def _test_n_step_forward_with_tuple_input(self, gpu):
        in_size = 5
        out_size = 3

        def concat_input(*args):
            return F.concat(args, axis=1)

        rseq = StatelessRecurrentSequential(
            concat_input,
            L.NStepRNNTanh(1, in_size, out_size, 0),
        )

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            rseq.to_gpu()
        xp = rseq.xp

        # Input is list of tuples. Each tuple has two variables.
        seqs_x = [
            (xp.random.uniform(-1, 1, size=(3, 2)).astype(np.float32),
             xp.random.uniform(-1, 1, size=(3, 3)).astype(np.float32)),
            (xp.random.uniform(-1, 1, size=(1, 2)).astype(np.float32),
             xp.random.uniform(-1, 1, size=(1, 3)).astype(np.float32)),
        ]

        # Concatenated output should be a variable.
        concat_out, concat_state = rseq.n_step_forward(
            seqs_x, None, output_mode='concat')
        self.assertEqual(concat_out.shape, (4, out_size))

        # Split output should be a list of variables.
        split_out, split_state = rseq.n_step_forward(
            seqs_x, None, output_mode='split')
        self.assertIsInstance(split_out, list)
        self.assertEqual(len(split_out), len(seqs_x))
        for seq_x, seq_out in zip(seqs_x, split_out):
            self.assertEqual(seq_out.shape, (len(seq_x), out_size))

        # Check if output_mode='concat' and output_mode='split' are consistent
        xp.testing.assert_allclose(
            F.concat(split_out, axis=0).array,
            concat_out.array,
        )

    @testing.attr.gpu
    def test_n_step_forward_with_tuple_input_gpu(self):
        self._test_n_step_forward_with_tuple_input(gpu=0)

    def test_n_step_forward_with_tuple_input_cpu(self):
        self._test_n_step_forward_with_tuple_input(gpu=-1)

    def _test_n_step_forward_with_tuple_output(self, gpu):
        in_size = 5
        out_size = 6

        def split_output(x):
            return tuple(F.split_axis(x, [2, 3], axis=1))

        rseq = StatelessRecurrentSequential(
            L.NStepRNNTanh(1, in_size, out_size, 0),
            split_output,
        )

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            rseq.to_gpu()
        xp = rseq.xp

        # Input is a list of two variables.
        seqs_x = [
            xp.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
            xp.random.uniform(-1, 1, size=(2, in_size)).astype(np.float32),
        ]

        # Concatenated output should be a tuple of three variables.
        concat_out, concat_state = rseq.n_step_forward(
            seqs_x, None, output_mode='concat')
        self.assertIsInstance(concat_out, tuple)
        self.assertEqual(len(concat_out), 3)
        self.assertEqual(concat_out[0].shape, (5, 2))
        self.assertEqual(concat_out[1].shape, (5, 1))
        self.assertEqual(concat_out[2].shape, (5, 3))

        # Split output should be a list of two tuples, each of which is of
        # three variables.
        split_out, split_state = rseq.n_step_forward(
            seqs_x, None, output_mode='split')
        self.assertIsInstance(split_out, list)
        self.assertEqual(len(split_out), 2)
        self.assertIsInstance(split_out[0], tuple)
        self.assertIsInstance(split_out[1], tuple)
        for seq_x, seq_out in zip(seqs_x, split_out):
            self.assertEqual(len(seq_out), 3)
            self.assertEqual(seq_out[0].shape, (len(seq_x), 2))
            self.assertEqual(seq_out[1].shape, (len(seq_x), 1))
            self.assertEqual(seq_out[2].shape, (len(seq_x), 3))

        # Check if output_mode='concat' and output_mode='split' are consistent
        xp.testing.assert_allclose(
            F.concat([F.concat(seq_out, axis=1)
                      for seq_out in split_out], axis=0).array,
            F.concat(concat_out, axis=1).array,
        )

    @testing.attr.gpu
    def test_n_step_forward_with_tuple_output_gpu(self):
        self._test_n_step_forward_with_tuple_output(gpu=0)

    def test_n_step_forward_with_tuple_output_cpu(self):
        self._test_n_step_forward_with_tuple_output(gpu=-1)

    def _test_mask_recurrent_state_at(self, gpu):
        in_size = 2
        out_size = 4
        rseq = StatelessRecurrentSequential(
            L.Linear(in_size, 3),
            F.elu,
            L.NStepGRU(1, 3, out_size, 0),
            F.softmax,
        )
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            rseq.to_gpu()
        xp = rseq.xp
        seqs_x = [
            xp.random.uniform(-1, 1, size=(2, in_size)).astype(np.float32),
            xp.random.uniform(-1, 1, size=(2, in_size)).astype(np.float32),
        ]
        transposed_x = F.transpose_sequence(seqs_x)
        print('transposed_x[0]', transposed_x[0])

        def no_mask_n_step_forward():
            nomask_nstep_out, nstep_rs = rseq.n_step_forward(
                seqs_x, None, output_mode='concat')
            return F.reshape(nomask_nstep_out, (2, 2, out_size)), nstep_rs
        nstep_out, nstep_rs = no_mask_n_step_forward()

        # Check if n_step_forward and forward twice results are same
        def no_mask_forward_twice():
            _, rs = rseq(transposed_x[0], None)
            return rseq(transposed_x[1], rs)
        nomask_out, nomask_rs = no_mask_forward_twice()
        xp.testing.assert_allclose(
            nstep_out.array[:, 1],
            nomask_out.array,
        )
        xp.testing.assert_allclose(nstep_rs[0].array, nomask_rs[0].array)

        # 1st-only mask forward twice: only 2nd should be the same
        def mask0_forward_twice():
            _, rs = rseq(transposed_x[0], None)
            rs = rseq.mask_recurrent_state_at(rs, 0)
            return rseq(transposed_x[1], rs)
        mask0_out, mask0_rs = mask0_forward_twice()
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out.array[0, 1],
                mask0_out.array[0],
            )
        xp.testing.assert_allclose(
            nstep_out.array[1, 1],
            mask0_out.array[1],
        )

        # 2nd-only mask forward twice: only 1st should be the same
        def mask1_forward_twice():
            _, rs = rseq(transposed_x[0], None)
            rs = rseq.mask_recurrent_state_at(rs, 1)
            return rseq(transposed_x[1], rs)
        mask1_out, mask1_rs = mask1_forward_twice()
        xp.testing.assert_allclose(
            nstep_out.array[0, 1],
            mask1_out.array[0],
        )
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out.array[1, 1],
                mask1_out.array[1],
            )

        # both 1st and 2nd mask forward twice: both should be different
        def mask01_forward_twice():
            _, rs = rseq(transposed_x[0], None)
            rs = rseq.mask_recurrent_state_at(rs, [0, 1])
            return rseq(transposed_x[1], rs)
        mask01_out, mask01_rs = mask01_forward_twice()
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out.array[0, 1],
                mask01_out.array[0],
            )
        with self.assertRaises(AssertionError):
            xp.testing.assert_allclose(
                nstep_out.array[1, 1],
                mask01_out.array[1],
            )

        # get and concat recurrent states and resume forward
        def get_and_concat_rs_forward():
            _, rs = rseq(transposed_x[0], None)
            rs0 = rseq.get_recurrent_state_at(rs, 0, unwrap_variable=True)
            rs1 = rseq.get_recurrent_state_at(rs, 1, unwrap_variable=True)
            concat_rs = rseq.concatenate_recurrent_states([rs0, rs1])
            return rseq(transposed_x[1], concat_rs)
        getcon_out, getcon_rs = get_and_concat_rs_forward()
        xp.testing.assert_allclose(getcon_rs[0].array, nomask_rs[0].array)
        xp.testing.assert_allclose(
            nstep_out.array[0, 1], getcon_out.array[0])
        xp.testing.assert_allclose(
            nstep_out.array[1, 1], getcon_out.array[1])

    @testing.attr.gpu
    def test_mask_recurrent_state_at_gpu(self):
        self._test_mask_recurrent_state_at(gpu=0)

    def test_mask_recurrent_state_at_cpu(self):
        self._test_mask_recurrent_state_at(gpu=-1)

    def _test_three_recurrent_children(self, gpu):
        # Test if https://github.com/chainer/chainer/issues/6053 is addressed
        in_size = 2
        out_size = 6

        rseq = StatelessRecurrentSequential(
            L.NStepLSTM(1, in_size, 3, 0),
            L.NStepGRU(2, 3, 4, 0),
            L.NStepRNNTanh(5, 4, out_size, 0),
        )

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            rseq.to_gpu()
        xp = rseq.xp

        seqs_x = [
            xp.random.uniform(-1, 1, size=(4, in_size)).astype(np.float32),
            xp.random.uniform(-1, 1, size=(1, in_size)).astype(np.float32),
            xp.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
        ]

        # Make and load a recurrent state to check if the order is correct.
        _, rs = rseq.n_step_forward(seqs_x, None, output_mode='concat')
        _, _ = rseq.n_step_forward(seqs_x, rs, output_mode='concat')

        _, rs = rseq.n_step_forward(seqs_x, None, output_mode='split')
        _, _ = rseq.n_step_forward(seqs_x, rs, output_mode='split')

    @testing.attr.gpu
    def test_three_recurrent_children_gpu(self):
        self._test_mask_recurrent_state_at(gpu=0)

    def test_three_recurrent_children_cpu(self):
        self._test_mask_recurrent_state_at(gpu=-1)
