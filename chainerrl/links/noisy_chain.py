import chainer
import chainer.functions as F

from chainerrl.initializers import LeCunNormal
from chainerrl.links.noisy_linear import FactorizedNoisyLinear


class NoisyMLP(chainer.Chain):
    """Noisy Networks

    See http://arxiv.org/abs/1706.10295
    """

    def __init__(self, in_size, out_size, hidden_sizes, nonlinearity=F.relu,
                 last_wscale=1):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity

        super().__init__()
        with self.init_scope():
            if hidden_sizes:
                hidden_layers = []
                hidden_layers.append(
                    FactorizedNoisyLinear(in_size, hidden_sizes[0]))
                for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                    hidden_layers.append(
                        FactorizedNoisyLinear(hin, hout))
                self.hidden_layers = chainer.ChainList(*hidden_layers)
                self.output = FactorizedNoisyLinear(
                    hidden_sizes[-1], out_size,
                    initialW=LeCunNormal(last_wscale))
            else:
                self.output = FactorizedNoisyLinear(
                    in_size, out_size,
                    initialW=LeCunNormal(last_wscale))

    def __call__(self, x):
        h = x
        if self.hidden_sizes:
            for l in self.hidden_layers:
                h = self.nonlinearity(l(h))
        return self.output(h)
