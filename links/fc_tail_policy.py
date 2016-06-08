import chainer
from chainer import links as L

import policy
from links.wn_linear import WNLinear


class FCTailPolicy(chainer.ChainList, policy.SoftmaxPolicy):

    def __init__(self, head, head_output_size, n_actions=18):
        layers = [
            head.copy(),
            L.Linear(head_output_size, n_actions),
        ]
        super(FCTailPolicy, self).__init__(*layers)

    def forward(self, state):
        h = self[0](state)
        return self[1](h)


class WNFCTailPolicy(chainer.ChainList, policy.SoftmaxPolicy):

    def __init__(self, head, head_output_size, n_actions=18):
        layers = [
            head.copy(),
            WNLinear(head_output_size, n_actions),
        ]
        super(WNFCTailPolicy, self).__init__(*layers)

    def forward(self, state):
        h = self[0](state)
        return self[1](h)
