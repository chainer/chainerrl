from chainerrl import links
from chainer import links as L

import chainer
from chainer.links.normalization.layer_normalization import LayerNormalization
from chainerrl.action_value import DiscreteActionValue
from chainerrl.action_value import DiscreteActionValueWithSigma
from chainer import functions as F

class MySequence(chainer.Chain):#links.Sequence):
    def __init__(self, obs, acts, head=False):
        """
        if head:
            super().__init__(
                L.Linear(obs, 32),
                F.relu,
                L.Linear(32, 32),
                F.relu,
                L.Linear(32, acts*2),
            )
        else:
            super().__init__(
                #links.MLP(obs, 32, [32]),
                L.Linear(obs, 32),
                F.relu,
                #LayerNormalization(),
                L.Linear(32, 32),
                F.relu,
                #LayerNormalization(),
                L.Linear(32, acts),
                DiscreteActionValue)
        """
        super().__init__()
        self.head = head
        self.acts = acts

        with self.init_scope():
            self.l1 = L.Linear(obs, 16)
            self.l2 = L.Linear(16, 16)
            self.l3 = L.Linear(16, acts*2 if head else acts)

        #self.add_link(self.l1)
        #self.add_link(self.l2)
        #self.add_link(self.l3)

    def scale_noise_coef(self, scale):
        try:
            self.l1.noise_coef *= scale
            self.l2.noise_coef *= scale
            self.l3.noise_coef *= scale
        except:
            pass

    def reset_noise(self):
        try:
            print("NOISE: ", self.layers[0].noise_coef)
            self.l1.reset_noise()
            self.l2.reset_noise()
            self.l3.reset_noise()
        except:
            pass

    def __call__(self, x, **kwargs):
        links = self.children()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        if self.head:
            return DiscreteActionValueWithSigma(x[:, :self.acts], x[:, self.acts:])
        else:

            return DiscreteActionValue(x)
