import chainer
from chainer import functions as F
from chainer import links as L


class VFunction(object):
    pass


class FCVFunction(chainer.ChainList, VFunction):

    def __init__(self, n_input_channels, n_hidden_layers=0,
                 n_hidden_channels=None):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels

        layers = []
        if n_hidden_layers > 0:
            layers.append(L.Linear(n_input_channels, n_hidden_channels))
            for i in range(n_hidden_layers - 1):
                layers.append(L.Linear(n_hidden_channels, n_hidden_channels))
            layers.append(L.Linear(n_hidden_channels, 1))
        else:
            layers.append(L.Linear(n_input_channels, 1))

        super(FCVFunction, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self[:-1]:
            h = F.relu(layer(h))
        h = self[-1](h)
        return h
