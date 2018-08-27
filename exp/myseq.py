from chainerrl import links
from chainer import links as L

import chainer
from chainer.links.normalization.layer_normalization import LayerNormalization
from chainerrl.action_value import DiscreteActionValue
from chainerrl.action_value import DiscreteActionValueWithSigma
from chainer import functions as F
from chainerrl import links
import numpy as np

class MySequence(chainer.Chain):#links.Sequence):
    def __init__(self, obs, acts, head=False, mean=1):
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
        self.mean = mean

        if obs is None:
            with self.init_scope():
                self.q_func = links.Sequence(
                    links.NatureDQNHead(activation=F.relu),
                    L.Linear(512, acts*(1+self.mean) if head else acts))
        else:
            with self.init_scope():
                self.l1 = L.Linear(obs, 100)
                self.l2 = L.Linear(100, 100)
                self.l3 = L.Linear(100, acts*(1+self.mean) if head else acts)

        #self.add_link(self.l1)
        #self.add_link(self.l2)
        #self.add_link(self.l3)

        self.obs = obs

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
        if self.obs is None:
            x = self.q_func(x)
        else:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = self.l3(x)

        if self.head:
            qval = x[:, :self.acts]

            if self.mean > 1:
                if True:#kwargs['avg']:
                    sigma = x[:, self.acts:]
                    sigma = F.reshape(sigma, (x.shape[0], self.mean, self.acts))
                    sigma = F.mean(sigma, axis=1)
                    return DiscreteActionValueWithSigma(qval, sigma)
                else:
                    batch_size = x.shape[0]
                    sigma_i = np.random.randint(0, self.mean)#, size=batch_size)
                    #sigma_i = np.repeat(sigma_i, self.acts)
                    #b = np.arange(batch_size)
                    sigma = x[:, self.acts*(sigma_i+1):self.acts*(sigma_i+2)]
                    return DiscreteActionValueWithSigma(qval, sigma)
            else:
                return DiscreteActionValueWithSigma(qval, x[:, self.acts:])
        else:
            return DiscreteActionValue(x)
