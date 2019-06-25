from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

from chainer import links as L
from chainer import testing
import numpy as np

from chainerrl.links import stateless_recurrent


class TestSplitOneStepBatchInput(unittest.TestCase):

    def test_array_input(self):
        # Input: an 4-length array
        # Expected output: a list of 4 arrays
        in_size = 3
        xs = np.random.uniform(-1, 1, size=(4, in_size)).astype(np.float32)
        split = stateless_recurrent.split_one_step_batch_input(xs)
        self.assertEqual(len(split), 4)
        np.testing.assert_allclose(split[0].array, [xs[0]])
        np.testing.assert_allclose(split[1].array, [xs[1]])
        np.testing.assert_allclose(split[2].array, [xs[2]])
        np.testing.assert_allclose(split[3].array, [xs[3]])

    def test_tuple_input(self):
        # Input: a tuple of 2 4-length arrays
        # Expected output: a list of 4 tuples, each of which has 2 arrays
        in_size0 = 2
        in_size1 = 3
        xs = (
            np.random.uniform(-1, 1, size=(4, in_size0)).astype(np.float32),
            np.random.uniform(-1, 1, size=(4, in_size1)).astype(np.float32),
        )
        split = stateless_recurrent.split_one_step_batch_input(xs)
        self.assertEqual(len(split), 4)
        self.assertEqual(len(split[0]), 2)
        np.testing.assert_allclose(split[0][0].array, [xs[0][0]])
        np.testing.assert_allclose(split[0][1].array, [xs[1][0]])
        np.testing.assert_allclose(split[1][0].array, [xs[0][1]])
        np.testing.assert_allclose(split[1][1].array, [xs[1][1]])
        np.testing.assert_allclose(split[2][0].array, [xs[0][2]])
        np.testing.assert_allclose(split[2][1].array, [xs[1][2]])
        np.testing.assert_allclose(split[3][0].array, [xs[0][3]])
        np.testing.assert_allclose(split[3][1].array, [xs[1][3]])


class TestSplitBatchedSequences(unittest.TestCase):

    def test_array_input(self):
        # Input: an 4+1+3=8-length array
        # Expected output: a list of 3 arrays (4-, 1-, and 3-length)
        in_size = 2
        seqs_x = [
            np.random.uniform(-1, 1, size=(4, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
        ]
        batched_seqs = np.concatenate(seqs_x, axis=0)
        self.assertEqual(batched_seqs.shape, (8, in_size))
        sections = [4, 5]
        split = stateless_recurrent.split_batched_sequences(
            batched_seqs, sections)
        self.assertEqual(len(split), 3)
        np.testing.assert_allclose(split[0].array, seqs_x[0])
        np.testing.assert_allclose(split[1].array, seqs_x[1])
        np.testing.assert_allclose(split[2].array, seqs_x[2])

    def test_tuple_input(self):
        # Input: a tuple of two 4+1+3=8-length arrays
        # Expected output: a list of 3 tuples, each has two arrays
        in_size0 = 2
        in_size1 = 3
        seqs_x0 = [
            np.random.uniform(-1, 1, size=(4, in_size0)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size0)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size0)).astype(np.float32),
        ]
        seqs_x1 = [
            np.random.uniform(-1, 1, size=(4, in_size1)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size1)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size1)).astype(np.float32),
        ]
        batched_seqs_x0 = np.concatenate(seqs_x0, axis=0)
        batched_seqs_x1 = np.concatenate(seqs_x1, axis=0)
        sections = [4, 5]
        split = stateless_recurrent.split_batched_sequences(
            (batched_seqs_x0, batched_seqs_x1), sections)
        self.assertEqual(len(split), 3)
        self.assertIsInstance(split[0], tuple)
        self.assertEqual(len(split[0]), 2)
        np.testing.assert_allclose(split[0][0].array, seqs_x0[0])
        np.testing.assert_allclose(split[0][1].array, seqs_x1[0])
        np.testing.assert_allclose(split[1][0].array, seqs_x0[1])
        np.testing.assert_allclose(split[1][1].array, seqs_x1[1])
        np.testing.assert_allclose(split[2][0].array, seqs_x0[2])
        np.testing.assert_allclose(split[2][1].array, seqs_x1[2])


class TestConcatenateSequences(unittest.TestCase):

    def test_array_input(self):
        # Input: a list of 3 arrays (4-, 1-, and 3-length)
        # Expected output: an 4+1+3=8-length array
        in_size = 2
        seqs_x = [
            np.random.uniform(-1, 1, size=(4, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
        ]
        concat_seqs = stateless_recurrent.concatenate_sequences(seqs_x)
        self.assertEqual(concat_seqs.shape, (8, in_size))
        np.testing.assert_allclose(concat_seqs[:4].array, seqs_x[0])
        np.testing.assert_allclose(concat_seqs[4:5].array, seqs_x[1])
        np.testing.assert_allclose(concat_seqs[5:].array, seqs_x[2])

    def test_tuple_input(self):
        # Input: a list of 3 tuples, each has two arrays
        # Expected output: a tuple of two 4+1+3=8-length arrays
        in_size0 = 2
        in_size1 = 3
        seqs_x0 = [
            np.random.uniform(-1, 1, size=(4, in_size0)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size0)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size0)).astype(np.float32),
        ]
        seqs_x1 = [
            np.random.uniform(-1, 1, size=(4, in_size1)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size1)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size1)).astype(np.float32),
        ]
        seqs_x = [
            (seqs_x0[0], seqs_x1[0]),
            (seqs_x0[1], seqs_x1[1]),
            (seqs_x0[2], seqs_x1[2]),
        ]
        concat_seqs = stateless_recurrent.concatenate_sequences(seqs_x)
        self.assertIsInstance(concat_seqs, tuple)
        self.assertEqual(len(concat_seqs), 2)
        self.assertEqual(concat_seqs[0].shape, (8, in_size0))
        self.assertEqual(concat_seqs[1].shape, (8, in_size1))
        np.testing.assert_allclose(concat_seqs[0][:4].array, seqs_x0[0])
        np.testing.assert_allclose(concat_seqs[0][4:5].array, seqs_x0[1])
        np.testing.assert_allclose(concat_seqs[0][5:].array, seqs_x0[2])
        np.testing.assert_allclose(concat_seqs[1][:4].array, seqs_x1[0])
        np.testing.assert_allclose(concat_seqs[1][4:5].array, seqs_x1[1])
        np.testing.assert_allclose(concat_seqs[1][5:].array, seqs_x1[2])


class TestStatelessRecurrentChainList(unittest.TestCase):

    def test(self):
        # Check if it can properly detect recurrent child links
        link = stateless_recurrent.StatelessRecurrentChainList(
            L.Linear(3, 4),
            L.NStepLSTM(1, 3, 2, 0),
            L.Linear(4, 5),
            stateless_recurrent.StatelessRecurrentChainList(
                L.NStepRNNTanh(1, 2, 5, 0),
            ),
        )
        self.assertEqual(len(link.recurrent_children), 2)
        self.assertIs(link.recurrent_children[0], link[1])
        self.assertIs(link.recurrent_children[1], link[3])
        self.assertEqual(len(link.recurrent_children[1].recurrent_children), 1)
        self.assertIs(
            link.recurrent_children[1].recurrent_children[0], link[3][0])


class TestCallRecurrentLink(unittest.TestCase):

    def _test_lstm(self, gpu):
        in_size = 2
        out_size = 3
        seqs_x = [
            np.random.uniform(-1, 1, size=(4, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
        ]
        link = L.NStepLSTM(1, in_size, out_size, 0)

        # Forward twice: with None and non-None random states
        h0, c0, y0 = link(None, None, seqs_x)
        h1, c1, y1 = link(h0, c0, seqs_x)
        self.assertEqual(h0.shape, (1, 3, out_size))
        self.assertEqual(c0.shape, (1, 3, out_size))
        self.assertEqual(h1.shape, (1, 3, out_size))
        self.assertEqual(c1.shape, (1, 3, out_size))

        # Forward twice via call_recurrent_link
        call_y0, call_rs0 = stateless_recurrent.call_recurrent_link(
            link, seqs_x, None, output_mode='split')
        self.assertEqual(len(call_y0), 3)
        np.testing.assert_allclose(call_y0[0].array, y0[0].array)
        np.testing.assert_allclose(call_y0[1].array, y0[1].array)
        np.testing.assert_allclose(call_y0[2].array, y0[2].array)
        self.assertEqual(len(call_rs0), 2)
        np.testing.assert_allclose(call_rs0[0].array, h0.array)
        np.testing.assert_allclose(call_rs0[1].array, c0.array)

        call_y1, call_rs1 = stateless_recurrent.call_recurrent_link(
            link, seqs_x, call_rs0, output_mode='split')
        self.assertEqual(len(call_y1), 3)
        np.testing.assert_allclose(call_y1[0].array, y1[0].array)
        np.testing.assert_allclose(call_y1[1].array, y1[1].array)
        np.testing.assert_allclose(call_y1[2].array, y1[2].array)
        self.assertEqual(len(call_rs1), 2)
        np.testing.assert_allclose(call_rs1[0].array, h1.array)
        np.testing.assert_allclose(call_rs1[1].array, c1.array)

        # Try output_mode=='concat' too
        concat_call_y0, concat_call_rs0 =\
            stateless_recurrent.call_recurrent_link(
                link, seqs_x, None, output_mode='concat')
        np.testing.assert_allclose(concat_call_y0[:4].array, y0[0].array)
        np.testing.assert_allclose(concat_call_y0[4:5].array, y0[1].array)
        np.testing.assert_allclose(concat_call_y0[5:].array, y0[2].array)
        self.assertEqual(len(concat_call_rs0), 2)
        np.testing.assert_allclose(concat_call_rs0[0].array, h0.array)
        np.testing.assert_allclose(concat_call_rs0[1].array, c0.array)

    @testing.attr.gpu
    def test_lstm_gpu(self):
        self._test_lstm(gpu=0)

    def test_lstm_cpu(self):
        self._test_lstm(gpu=-1)

    def _test_non_lstm(self, gpu, name):
        in_size = 2
        out_size = 3
        seqs_x = [
            np.random.uniform(-1, 1, size=(4, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
        ]
        self.assertTrue(name in ('NStepGRU', 'NStepRNNReLU', 'NStepRNNTanh'))
        cls = getattr(L, name)
        link = cls(1, in_size, out_size, 0)

        # Forward twice: with None and non-None random states
        h0, y0 = link(None, seqs_x)
        h1, y1 = link(h0, seqs_x)
        self.assertEqual(h0.shape, (1, 3, out_size))
        self.assertEqual(h1.shape, (1, 3, out_size))

        # Forward twice via call_recurrent_link
        call_y0, call_rs0 = stateless_recurrent.call_recurrent_link(
            link, seqs_x, None, output_mode='split')
        self.assertEqual(len(call_y0), 3)
        np.testing.assert_allclose(call_y0[0].array, y0[0].array)
        np.testing.assert_allclose(call_y0[1].array, y0[1].array)
        np.testing.assert_allclose(call_y0[2].array, y0[2].array)
        np.testing.assert_allclose(call_rs0.array, h0.array)

        call_y1, call_rs1 = stateless_recurrent.call_recurrent_link(
            link, seqs_x, call_rs0, output_mode='split')
        self.assertEqual(len(call_y1), 3)
        np.testing.assert_allclose(call_y1[0].array, y1[0].array)
        np.testing.assert_allclose(call_y1[1].array, y1[1].array)
        np.testing.assert_allclose(call_y1[2].array, y1[2].array)
        np.testing.assert_allclose(call_rs1.array, h1.array)

        # Try output_mode=='concat' too
        concat_call_y0, concat_call_rs0 =\
            stateless_recurrent.call_recurrent_link(
                link, seqs_x, None, output_mode='concat')
        np.testing.assert_allclose(concat_call_y0[:4].array, y0[0].array)
        np.testing.assert_allclose(concat_call_y0[4:5].array, y0[1].array)
        np.testing.assert_allclose(concat_call_y0[5:].array, y0[2].array)
        np.testing.assert_allclose(concat_call_rs0.array, h0.array)

    @testing.attr.gpu
    def test_gru_gpu(self):
        self._test_non_lstm(gpu=0, name='NStepGRU')

    def test_gru_cpu(self):
        self._test_non_lstm(gpu=-1, name='NStepGRU')

    @testing.attr.gpu
    def test_rnn_relu_gpu(self):
        self._test_non_lstm(gpu=0, name='NStepRNNReLU')

    def test_rnn_relu_cpu(self):
        self._test_non_lstm(gpu=-1, name='NStepRNNReLU')

    @testing.attr.gpu
    def test_rnn_tanh_gpu(self):
        self._test_non_lstm(gpu=0, name='NStepRNNTanh')

    def test_rnn_tanh_cpu(self):
        self._test_non_lstm(gpu=-1, name='NStepRNNTanh')


class TestRecurrentStateFunctions(unittest.TestCase):

    def _test_lstm(self, gpu):
        in_size = 2
        out_size = 3
        seqs_x = [
            np.random.uniform(-1, 1, size=(4, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
        ]
        link = L.NStepLSTM(1, in_size, out_size, 0)

        # Forward twice: with None and non-None random states
        h0, c0, y0 = link(None, None, seqs_x)
        h1, c1, y1 = link(h0, c0, seqs_x)
        self.assertEqual(h0.shape, (1, 3, out_size))
        self.assertEqual(c0.shape, (1, 3, out_size))
        self.assertEqual(h1.shape, (1, 3, out_size))
        self.assertEqual(c1.shape, (1, 3, out_size))

        # Masked at 0
        rs0_mask0 = stateless_recurrent.mask_recurrent_state_at(
            link, (h0, c0), 0)
        _, _, y1m0 = link(rs0_mask0[0], rs0_mask0[1], seqs_x)
        np.testing.assert_allclose(y1m0[0].array, y0[0].array)
        np.testing.assert_allclose(y1m0[1].array, y1[1].array)
        np.testing.assert_allclose(y1m0[2].array, y1[2].array)

        # Masked at 1
        rs0_mask1 = stateless_recurrent.mask_recurrent_state_at(
            link, (h0, c0), 1)
        _, _, y1m1 = link(rs0_mask1[0], rs0_mask1[1], seqs_x)
        np.testing.assert_allclose(y1m1[0].array, y1[0].array)
        np.testing.assert_allclose(y1m1[1].array, y0[1].array)
        np.testing.assert_allclose(y1m1[2].array, y1[2].array)

        # Masked at (1, 2)
        rs0_mask12 = stateless_recurrent.mask_recurrent_state_at(
            link, (h0, c0), (1, 2))
        _, _, y1m12 = link(rs0_mask12[0], rs0_mask12[1], seqs_x)
        np.testing.assert_allclose(y1m12[0].array, y1[0].array)
        np.testing.assert_allclose(y1m12[1].array, y0[1].array)
        np.testing.assert_allclose(y1m12[2].array, y0[2].array)

        # Get at 1 and concat with None
        rs0_get1 = stateless_recurrent.get_recurrent_state_at(
            link, (h0, c0), 1, unwrap_variable=False)
        np.testing.assert_allclose(rs0_get1[0].array, h0.array[:, 1])
        np.testing.assert_allclose(rs0_get1[1].array, c0.array[:, 1])
        concat_rs_get1 = stateless_recurrent.concatenate_recurrent_states(
            link, [None, rs0_get1, None])
        _, _, y1g1 = link(concat_rs_get1[0], concat_rs_get1[1], seqs_x)
        np.testing.assert_allclose(y1g1[0].array, y0[0].array)
        np.testing.assert_allclose(y1g1[1].array, y1[1].array)
        np.testing.assert_allclose(y1g1[2].array, y0[2].array)

        # Get at 1 with unwrap_variable=True
        rs0_get1 = stateless_recurrent.get_recurrent_state_at(
            link, (h0, c0), 1, unwrap_variable=True)
        np.testing.assert_allclose(rs0_get1[0], h0.array[:, 1])
        np.testing.assert_allclose(rs0_get1[1], c0.array[:, 1])

    @testing.attr.gpu
    def test_lstm_gpu(self):
        self._test_lstm(gpu=0)

    def test_lstm_cpu(self):
        self._test_lstm(gpu=-1)

    def _test_non_lstm(self, gpu, name):
        in_size = 2
        out_size = 3
        seqs_x = [
            np.random.uniform(-1, 1, size=(4, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(1, in_size)).astype(np.float32),
            np.random.uniform(-1, 1, size=(3, in_size)).astype(np.float32),
        ]
        self.assertTrue(name in ('NStepGRU', 'NStepRNNReLU', 'NStepRNNTanh'))
        cls = getattr(L, name)
        link = cls(1, in_size, out_size, 0)

        # Forward twice: with None and non-None random states
        h0, y0 = link(None, seqs_x)
        h1, y1 = link(h0, seqs_x)
        self.assertEqual(h0.shape, (1, 3, out_size))
        self.assertEqual(h1.shape, (1, 3, out_size))

        # Masked at 0
        rs0_mask0 = stateless_recurrent.mask_recurrent_state_at(link, h0, 0)
        _, y1m0 = link(rs0_mask0, seqs_x)
        np.testing.assert_allclose(y1m0[0].array, y0[0].array)
        np.testing.assert_allclose(y1m0[1].array, y1[1].array)
        np.testing.assert_allclose(y1m0[2].array, y1[2].array)

        # Masked at (1, 2)
        rs0_mask12 = stateless_recurrent.mask_recurrent_state_at(
            link, h0, (1, 2))
        _, y1m12 = link(rs0_mask12, seqs_x)
        np.testing.assert_allclose(y1m12[0].array, y1[0].array)
        np.testing.assert_allclose(y1m12[1].array, y0[1].array)
        np.testing.assert_allclose(y1m12[2].array, y0[2].array)

        # Get at 1 and concat with None
        rs0_get1 = stateless_recurrent.get_recurrent_state_at(
            link, h0, 1, unwrap_variable=False)
        np.testing.assert_allclose(rs0_get1.array, h0.array[:, 1])
        concat_rs_get1 = stateless_recurrent.concatenate_recurrent_states(
            link, [None, rs0_get1, None])
        _, y1g1 = link(concat_rs_get1, seqs_x)
        np.testing.assert_allclose(y1g1[0].array, y0[0].array)
        np.testing.assert_allclose(y1g1[1].array, y1[1].array)
        np.testing.assert_allclose(y1g1[2].array, y0[2].array)

        # Get at 1 with unwrap_variable=True
        rs0_get1 = stateless_recurrent.get_recurrent_state_at(
            link, h0, 1, unwrap_variable=True)
        np.testing.assert_allclose(rs0_get1, h0.array[:, 1])

    @testing.attr.gpu
    def test_gru_gpu(self):
        self._test_non_lstm(gpu=0, name='NStepGRU')

    def test_gru_cpu(self):
        self._test_non_lstm(gpu=-1, name='NStepGRU')

    @testing.attr.gpu
    def test_rnn_relu_gpu(self):
        self._test_non_lstm(gpu=0, name='NStepRNNReLU')

    def test_rnn_relu_cpu(self):
        self._test_non_lstm(gpu=-1, name='NStepRNNReLU')

    @testing.attr.gpu
    def test_rnn_tanh_gpu(self):
        self._test_non_lstm(gpu=0, name='NStepRNNTanh')

    def test_rnn_tanh_cpu(self):
        self._test_non_lstm(gpu=-1, name='NStepRNNTanh')
