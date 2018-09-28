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

        self.sigma_nets = []

        if obs is None:
            with self.init_scope():
                self.q_func = links.Sequence(
                    links.NatureDQNHead(activation=F.relu),
                    L.Linear(512, acts))

                for n in range(mean):
                    q_func = links.Sequence(
                        links.NatureDQNHead(activation=F.relu),
                        L.Linear(512, acts))
                    self.sigma_nets.append(q_func)
        else:
            with self.init_scope():
                self.l1 = L.Linear(obs, 100)
                self.l2 = L.Linear(100, 100)
                self.l3 = L.Linear(100, acts)

            for n in range(mean):
                nl1 = L.Linear(obs, 100)
                nl2 = L.Linear(100, 100)
                nl3 = L.Linear(100, acts)

                self.add_link(str(n)+'_l1', nl1)
                self.add_link(str(n)+'_l2', nl2)
                self.add_link(str(n)+'_l3', nl3)

                """
                def net(x):
                    x = F.relu(nl1(x))
                    x = F.relu(nl2(x))
                    x = nl3(x)
                    return x
                """
                net = [nl1, nl2, nl3]

                self.sigma_nets.append(net)

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

    def __call__(self, input, **kwargs):
        if self.obs is None:
            x = self.q_func(input)
        else:
            x = F.relu(self.l1(input))
            x = F.relu(self.l2(x))
            x = self.l3(x)

        if self.head:
            if self.mean > 1:
                #sigma = x[:, self.acts:]
                #sigma = F.reshape(sigma, (x.shape[0], self.mean, self.acts))
                sigmas = []
                for i in range(self.mean):
                    l1, l2, l3 = self.sigma_nets[i]
                    x = F.relu(l1(input))
                    x = F.relu(l2(x))
                    x = l3(x)
                    sigmas.append(x)
                sigmas = F.stack(sigmas, axis=0)
                sigmas = F.softplus(sigmas)
                sigma = F.mean(sigmas, axis=0)
                return DiscreteActionValueWithSigma(x, sigma, all_sigmas=sigmas)
            else:
                return DiscreteActionValueWithSigma(x, F.softplus(self.sigma_nets[0](input)))
        else:
            return DiscreteActionValue(x)
