from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from cached_property import cached_property
import chainer
import numpy as np

from chainerrl.links.stateless_recurrent import call_recurrent_link
from chainerrl.links.stateless_recurrent import concatenate_sequences
from chainerrl.links.stateless_recurrent import is_recurrent_link
from chainerrl.links.stateless_recurrent import split_batched_sequences
from chainerrl.links.stateless_recurrent import StatelessRecurrentChainList


class StatelessRecurrentSequential(
        StatelessRecurrentChainList, chainer.Sequential):
    """Sequential model that can contain stateless recurrent links.

    This link a stateless recurrent analog to chainer.Sequential. It supports
    the stateless recurrent interface by automatically detecting recurrent
    links and handles recurrent states properly.

    For non-recurrent layers (non-link callables or non-recurrent callable
    links), this link automatically concatenates the input to the layers
    for efficient computation.

    Args:
        *layers: Callable objects.
    """

    def n_step_forward(self, sequences, recurrent_state, output_mode):
        assert sequences
        assert output_mode in ['split', 'concat']
        if recurrent_state is None:
            recurrent_state_queue = [None] * len(self.recurrent_children)
        else:
            assert len(recurrent_state) == len(self.recurrent_children)
            recurrent_state_queue = list(reversed(recurrent_state))
        new_recurrent_state = []
        h = sequences
        seq_mode = True
        sections = np.cumsum([len(x) for x in sequences[:-1]], dtype=np.int32)
        for layer in self._layers:
            if is_recurrent_link(layer):
                if not seq_mode:
                    h = split_batched_sequences(h, sections)
                    seq_mode = True
                rs = recurrent_state_queue.pop()
                h, rs = call_recurrent_link(layer, h, rs, output_mode='split')
                new_recurrent_state.append(rs)
            else:
                if seq_mode:
                    seq_mode = False
                    h = concatenate_sequences(h)
                if isinstance(h, tuple):
                    h = layer(*h)
                else:
                    h = layer(h)
        if not seq_mode and output_mode == 'split':
            h = split_batched_sequences(h, sections)
            seq_mode = True
        elif seq_mode and output_mode == 'concat':
            h = concatenate_sequences(h)
            seq_mode = False
        assert seq_mode is (output_mode == 'split')
        assert not recurrent_state_queue
        assert len(new_recurrent_state) == len(self.recurrent_children)
        return h, tuple(new_recurrent_state)

    @cached_property
    def recurrent_children(self):
        """Return recurrent child links.

        This overrides `StatelessRecurrentChainList.recurrent_children`
        because `Sequential`'s evaluation order can be different from the
        order of links in `Sequential.children()`.

        See https://github.com/chainer/chainer/issues/6053

        Returns:
            tuple: Tuple of `chainer.Link`s that are recurrent.
        """
        return tuple(child for child in self._layers
                     if is_recurrent_link(child))
