from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import super
from future import standard_library
standard_library.install_aliases()
import chainer
from chainer import links as L

import v_function
from links.wn_linear import WNLinear


class FCTailVFunction(chainer.ChainList, v_function.VFunction):

    def __init__(self, head, head_output_size):

        layers = [
            head.copy(),
            L.Linear(head_output_size, 1),
        ]

        super(FCTailVFunction, self).__init__(*layers)

    def __call__(self, state):
        h = self[0](state)
        return self[1](h)


class WNFCTailVFunction(chainer.ChainList, v_function.VFunction):

    def __init__(self, head, head_output_size):

        layers = [
            head.copy(),
            WNLinear(head_output_size, 1),
        ]

        super(WNFCTailVFunction, self).__init__(*layers)

    def __call__(self, state):
        h = self[0](state)
        return self[1](h)
