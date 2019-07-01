from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from cached_property import cached_property
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


def split_one_step_batch_input(xs):
    """Split one-step batch input.

    Args:
        xs (chainer.Variable, ndarray or tuple): One-step batched input. It
            should be either:
                - a variable whose first axis is the batch axis.
                - a tuple of such variables.

    Returns:
        list: Either a list of variables or a list of tuples of varialbes.
            The length of the list is the batch size of the input.
    """
    if isinstance(xs, tuple):
        return list(zip(*[split_one_step_batch_input(x) for x in xs]))
    else:
        return list(F.split_axis(xs, len(xs), axis=0))


class StatelessRecurrent(object):
    """Stateless recurrent link interface.

    This class defines the interface of a recurrent link ChainerRL can handle.

    In most casese, you can just use ChainerRL's existing containers like
    `chainerrl.links.StatelessRecurrentChainList`,
    `chainerrl.links.StatelessRecurrentSequential`, and
    `chainerrl.links.StatelessRecurrentBranched` to define a recurrent
    link. You can use Chainer's recurrent links such as L.NStepLSTM inside the
    containers.

    To write your own recurrent link, you need to implement the interface.
    """

    def n_step_forward(self, x, recurrent_state):
        """Multi-step batch forward computation.

        This method sequentially applies layers as chainer.Sequential does.

        Args:
            x (list): Input sequences. Each sequence should be a variable whose
                first axis corresponds to time or a tuple of such variables.
            recurrent_state (object): Batched recurrent state. If set to None,
                it is initialized.
            output_mode (str): If set to 'concat', the output value is
                concatenated into a single large batch, which can be suitable
                for loss computation. If set to 'split', the output value is
                a list of output sequences.

        Returns:
            object: Output sequences. See the description of the `output_mode`
                argument.
            object: New batched recurrent state.
        """
        raise NotImplementedError

    def __call__(self, x, recurrent_state):
        """One-step batch forward computation.

        Args:
            x (chainer.Variable, ndarray, or tuple): One-step batched input.
            recurrent_state (object): Batched recurrent state.

        Returns:
            chainer.Variable, ndarray, or tuple: One-step batched output.
            object: New batched recurrent state.
        """
        assert isinstance(x, (chainer.Variable, self.xp.ndarray))
        return self.n_step_forward(
            split_one_step_batch_input(x),
            recurrent_state,
            output_mode='concat',
        )

    def mask_recurrent_state_at(self, recurrent_state, indices):
        """Return a recurrent state masked at given indices.

        This method can be used to initialize a recurrent state only for a
        certain sequence, not all the sequences.

        Args:
            recurrent_state (object): Batched recurrent state.
            indices (int or array-like of ints): Which recurrent state to mask.

        Returns:
            object: New batched recurrent state.
        """
        raise NotImplementedError

    def get_recurrent_state_at(self, recurrent_state, indices):
        """Get a recurrent state at given indices.

        This method can be used to save a recurrent state so that you can
        reuse it when you replay past sequences.

        Args:
            indices (int or array-like of ints): Which recurrent state to get.

        Returns:
            object: Recurrent state of given indices.
        """
        raise NotImplementedError

    def concatenate_recurrent_states(self, split_recurrent_states):
        """Concatenate recurrent states into a batch.

        This method can be used to make a batched recurrent state from separate
        recurrent states obtained via the `get_recurrent_state_at` method.

        Args:
            split_recurrent_states (object): Recurrent states to concatenate.

        Returns:
            object: Batched recurrent_state.
        """
        raise NotImplementedError


def is_recurrent_link(layer):
    """Return True iff a given layer is recurrent and supported by ChainerRL.

    Args:
        layer (callable): Any callable object.

    Returns:
        bool: True iff a given layer is recurrent and supported by ChainerRL.
    """
    return isinstance(layer, (
        L.NStepLSTM,
        L.NStepGRU,
        L.NStepRNNReLU,
        L.NStepRNNTanh,
        StatelessRecurrent,
    ))


def split_batched_sequences(xs, sections):
    """Split concatenated sequences.

    Args:
        xs (chainer.Variable, ndarray or tuple): Concatenated sequences.
        sections (array-like): Sections as indices indicating start positions
            of sequences.

    Returns:
        list: List of sequences.
    """
    if isinstance(xs, tuple):
        return list(zip(*[split_batched_sequences(x, sections) for x in xs]))
    else:
        return list(F.split_axis(xs, sections, axis=0))


def concatenate_sequences(sequences):
    """Concatenate sequences.

    Args:
        sequences (list): List of sequences. The following two cases are
            supported:
                - (a) Each sequence is a Variable or ndarray.
                - (b) Each sequence is tuple of a Variable or ndarray.

    Returns:
        chainer.Variable, ndarray or tuple: Concatenated sequences.
    """
    if isinstance(sequences[0], tuple):
        tuple_size = len(sequences[0])
        return tuple(
            F.concat([seq[i] for seq in sequences], axis=0)
            for i in range(tuple_size))
        raise NotImplementedError
    else:
        return F.concat(sequences, axis=0)


def call_recurrent_link(link, sequences, recurrent_state, output_mode):
    """Call a recurrent link following the interface of `StatelessRecurrent`.

    Args:
        link (chainer.Link): Recurrent link.
        sequences, recurrent_state, output_mode: See the docstring of
            `StatelessRecurrent.n_step_forward`.

    Returns:
        object: Output sequences. See the docstring of
            `StatelessRecurrent.n_step_forward`.
        object: New batched recurrent state.
    """
    assert isinstance(link, chainer.Link)
    assert isinstance(sequences, list)
    if isinstance(link, L.NStepLSTM):
        if recurrent_state is None:
            h = None
            c = None
        else:
            h, c = recurrent_state
        h, c, sequences = link(h, c, sequences)
        if output_mode == 'concat':
            sequences = concatenate_sequences(sequences)
        return sequences, (h, c)
    if isinstance(link, (L.NStepGRU, L.NStepRNNReLU, L.NStepRNNTanh)):
        h = recurrent_state
        h, sequences = link(h, sequences)
        if output_mode == 'concat':
            sequences = concatenate_sequences(sequences)
        return sequences, h
    if isinstance(link, StatelessRecurrent):
        return link.n_step_forward(
            sequences, recurrent_state, output_mode=output_mode)
    else:
        raise ValueError('{} is not a recurrent link'.format(link))


def mask_recurrent_states_of_links_at(links, recurrent_states, indices):
    if recurrent_states is None:
        return None
    assert len(links) == len(recurrent_states)
    return [mask_recurrent_state_at(link, rs, indices)
            for link, rs in zip(links, recurrent_states)]


def get_recurrent_states_of_links_at(
        links, recurrent_states, indices, unwrap_variable):
    if recurrent_states is None:
        return [None] * len(links)
    assert len(links) == len(recurrent_states)
    return [get_recurrent_state_at(link, rs, indices, unwrap_variable)
            for link, rs in zip(links, recurrent_states)]


def concatenate_recurrent_states_of_links(links, split_recurrent_states):
    assert split_recurrent_states is not None
    # Replace None with a list of None
    split_recurrent_states = list(split_recurrent_states)
    for i, srs in enumerate(split_recurrent_states):
        if srs is None:
            split_recurrent_states[i] = [None] * len(links)
        else:
            assert len(srs) == len(links)
    # Transpose first two axes of (batch_size, n_recurrent_links, ...)
    transposed = list(zip(*split_recurrent_states))
    assert len(links) == len(transposed)
    return [concatenate_recurrent_states(link, srs)
            for link, srs in zip(links, transposed)]


def mask_recurrent_state_at(link, recurrent_state, indices):
    if recurrent_state is None:
        return None
    if isinstance(link, L.NStepLSTM):
        h, c = recurrent_state
        # shape: (n_layers, batch_size, out_size)
        assert h.ndim == 3
        assert c.ndim == 3
        mask = link.xp.ones_like(h.array)
        mask[:, indices] = 0
        c = c * mask
        h = h * mask
        return (h, c)
    if isinstance(link, (L.NStepGRU, L.NStepRNNReLU, L.NStepRNNTanh)):
        h = recurrent_state
        # shape: (n_layers, batch_size, out_size)
        assert h.ndim == 3
        mask = link.xp.ones_like(h.array)
        mask[:, indices] = 0
        h = h * mask
        return h
    if isinstance(link, StatelessRecurrent):
        return link.mask_recurrent_state_at(recurrent_state, indices)
    else:
        raise ValueError('{} is not a recurrent link'.format(link))


def get_recurrent_state_at(link, recurrent_state, indices, unwrap_variable):
    if recurrent_state is None:
        return None
    if isinstance(link, L.NStepLSTM):
        h, c = recurrent_state
        if unwrap_variable:
            h = h.array
            c = c.array
        # shape: (n_layers, batch_size, out_size)
        assert h.ndim == 3
        assert c.ndim == 3
        return (h[:, indices], c[:, indices])
    if isinstance(link, (L.NStepGRU, L.NStepRNNReLU, L.NStepRNNTanh)):
        h = recurrent_state
        if unwrap_variable:
            h = h.array
        # shape: (n_layers, batch_size, out_size)
        assert h.ndim == 3
        return h[:, indices]
    if isinstance(link, StatelessRecurrent):
        return link.get_recurrent_state_at(
            recurrent_state, indices, unwrap_variable)
    else:
        raise ValueError('{} is not a recurrent link'.format(link))


def concatenate_recurrent_states(link, split_recurrent_states):
    if isinstance(link, L.NStepLSTM):
        # shape: (n_layers, batch_size, out_size)
        n_layers = link.n_layers
        out_size = link.out_size
        xp = link.xp
        hs = []
        cs = []
        for i, srs in enumerate(split_recurrent_states):
            if srs is None:
                h = xp.zeros((n_layers, 1, out_size), dtype=np.float32)
                c = xp.zeros((n_layers, 1, out_size), dtype=np.float32)
            else:
                h, c = srs
                if h.ndim == 2:
                    assert h.shape == (n_layers, out_size)
                    assert c.shape == (n_layers, out_size)
                    # add batch axis
                    h = h[:, None]
                    c = c[:, None]
            hs.append(h)
            cs.append(c)
        h = F.concat(hs, axis=1)
        c = F.concat(cs, axis=1)
        return (h, c)
    if isinstance(link, (L.NStepGRU, L.NStepRNNReLU, L.NStepRNNTanh)):
        n_layers = link.n_layers
        out_size = link.out_size
        xp = link.xp
        hs = []
        for i, srs in enumerate(split_recurrent_states):
            if srs is None:
                h = xp.zeros((n_layers, 1, out_size), dtype=np.float32)
            else:
                h = srs
                if h.ndim == 2:
                    assert h.shape == (n_layers, out_size)
                    # add batch axis
                    h = h[:, None]
            hs.append(h)
        h = F.concat(hs, axis=1)
        return h
    if isinstance(link, StatelessRecurrent):
        return link.concatenate_recurrent_states(split_recurrent_states)
    else:
        raise ValueError('{} is not a recurrent link'.format(link))


class StatelessRecurrentChainList(StatelessRecurrent, chainer.ChainList):
    """ChainList that auutomatically handles recurrent states.

    This link extends chainer.ChainList by adding the `recurrent_children`
    property that returns all the recurrent child links and implementing
    recurrent state manimulation methods required for the StatelessRecurrent
    interface.

    A recurrent state for this link is defined as a tuple of recurrent states
    of child recurrent links.
    """

    @cached_property
    def recurrent_children(self):
        """Return recurrent child links.

        Returns:
            tuple: Tuple of `chainer.Link`s that are recurrent.
        """
        return tuple(child for child in self.children()
                     if is_recurrent_link(child))

    def mask_recurrent_state_at(self, recurrent_states, indices):
        return mask_recurrent_states_of_links_at(
            self.recurrent_children, recurrent_states, indices)

    def get_recurrent_state_at(
            self, recurrent_states, indices, unwrap_variable):
        return get_recurrent_states_of_links_at(
            self.recurrent_children, recurrent_states, indices,
            unwrap_variable)

    def concatenate_recurrent_states(self, split_recurrent_states):
        return concatenate_recurrent_states_of_links(
            self.recurrent_children, split_recurrent_states)
