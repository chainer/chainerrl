from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer

from chainerrl.links.stateless_recurrent import call_recurrent_link
from chainerrl.links.stateless_recurrent import StatelessRecurrentChainList


class StatelessRecurrentBranched(
        StatelessRecurrentChainList, chainer.ChainList):
    """Stateless recurrent parallel link.

    This is a recurrent analog to chainerrl.links.Branched. It bundles
    multiple links that implements `StatelessRecurrent`.

    Args:
        *links: Child links. Each link should be recurrent and callable.
    """

    def n_step_forward(self, sequences, recurrent_state, output_mode):
        if recurrent_state is None:
            n = len(self)
            recurrent_state = [None] * n
        child_ys, rs = tuple(zip(*[
            call_recurrent_link(link, sequences, rs, output_mode)
            for link, rs in zip(self, recurrent_state)]))
        if output_mode == 'concat':
            return child_ys, rs
        assert output_mode == 'split'
        assert len(child_ys) == len(self)
        assert len(child_ys[0]) == len(sequences)
        assert len(rs) == len(self)
        out = list(zip(*child_ys))
        return out, rs
