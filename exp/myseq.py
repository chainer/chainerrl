from chainerrl import links
from chainer import links as L
from chainer.links.normalization.layer_normalization import LayerNormalization
from chainerrl.action_value import DiscreteActionValue

class MySequence(links.Sequence):
    def __init__(self, obs, acts):
        super().__init__(
            #links.MLP(obs, 32, [32]),
            L.Linear(obs, 32),
            #LayerNormalization(),
            L.Linear(32, 32),
            #LayerNormalization(),
            L.Linear(32, acts),
            DiscreteActionValue)

    def scale_noise_coef(self, scale):
        try:
            self.layers[0].noise_coef *= scale
            self.layers[2].noise_coef *= scale
            self.layers[4].noise_coef *= scale
        except:
            pass

    def reset_noise(self):
        try:
            print("NOISE: ", self.layers[0].noise_coef)
            self.layers[0].reset_noise()
            self.layers[2].reset_noise()
            self.layers[4].reset_noise()
        except:
            pass

    def __call__(self, x, **kwargs):
        return super().__call__(x, **kwargs)
