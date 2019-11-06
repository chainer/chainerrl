from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import gym
import numpy as np
import random

from chainerrl.misc.batch_states import batch_states

def subseq(seq, subseq_len, start):
    return seq[start: start + subseq_len]


class TREXNet(chainer.ChainList):
    """TREX's architecture: https://arxiv.org/abs/1904.06387"""


    def __init__(self):
        layers = [
            L.Convolution2D(4, 16, 7, stride=3),
            L.Convolution2D(16, 16, 5, stride=2),
            L.Convolution2D(16, 16, 3, stride=1),
            L.Convolution2D(16, 16, 3, stride=1),
            L.Linear(784, 64),
            L.Linear(64, 1)
        ]

        super(TREXNet, self).__init__(*layers)

    def __call__(self, trajectory):
        h = trajectory
        for layer in self:
            h = F.leaky_relu(layer(h))
        return h


class TREXReward(gym.Wrapper):
    """Implements Trajectory-ranked Reward EXtrapolation (TREX):

    https://arxiv.org/abs/1904.06387.

    Args:
        env: a ScoreMaskEnv
        ranked_demos (RankedDemoDataset): A list of ranked demonstrations
        steps: number of gradient steps
        sub_traj_len: a tuple containing (min, max) traj length to sample
        traj_batch_size: num trajectory pairs to use per update
        opt: optimizer
        network: A reward network to train

    Attributes:
        demos: A list of demonstrations
        trex_network: Reward network

    """

    def __init__(self, env,
                 ranked_demos,
                 steps=30000,
                 num_sub_trajs=12800,
                 sub_traj_len=(50,100),
                 traj_batch_size=16,
                 opt=optimizers.Adam(alpha=0.00005),
                 sample_live=True,
                 network=TREXNet(),
                 gpu=None,
                 save_network=False):
        super().__init__(env)
        self.ranked_demos = ranked_demos
        self.steps = steps
        self.trex_network = network
        self.opt = opt
        self.opt.setup(self.trex_network)
        self.training_observations = []
        self.training_labels = []
        self.prev_reward = None
        self.traj_batch_size = traj_batch_size
        self.min_sub_traj_len = sub_traj_len[0]
        self.max_sub_traj_len = sub_traj_len[1]
        self.num_sub_trajs = num_sub_trajs
        self.sample_live = sample_live
        self.save_network = save_network
        self.examples = []       
        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.trex_reward.to_gpu(device=gpu)
        self._train()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        obs = batch_states([observation], self.trex_network.xp, self._phi)
        with chainer.no_backprop_mode():
            trex_reward = F.sigmoid(self.trex_network(obs))
        info["true_reward"] = reward
        return observation, trex_reward, done, info

    def create_example(self):
        '''Creates a training example.'''

        ranked_trajs = self.ranked_demos.episodes
        indices = np.arange(len(ranked_trajs)).tolist()
        traj_indices = np.random.choice(indices, size=2, replace=False)
        i = traj_indices[0]
        j = traj_indices[1]
        min_ep_len = min(len(ranked_trajs[i]), len(ranked_trajs[j]))
        sub_traj_len = np.random.randint(self.min_sub_traj_len,
                                         self.max_sub_traj_len)
        traj_1 = ranked_trajs[i]
        traj_2 = ranked_trajs[j]
        if i < j:
            i_start = np.random.randint(min_ep_len - sub_traj_len + 1)
            j_start = np.random.randint(i_start, len(traj_2) - sub_traj_len + 1)
        else:
            j_start = np.random.randint(min_ep_len - sub_traj_len + 1)
            i_start = np.random.randint(j_start, len(traj_1) - sub_traj_len + 1)
        sub_traj_i = subseq(traj_1, sub_traj_len, start=i_start)
        sub_traj_j = subseq(traj_2, sub_traj_len, start=j_start)
        # if trajectory i is better than trajectory j
        if i > j:
            label = 0
        else:
            label = 1
        return sub_traj_i, sub_traj_j, label

    def create_training_dataset(self):
        self.examples = []
        self.index = 0
        for _ in range(self.num_sub_trajs):
            self.examples.append(self.create_example())


    def get_training_batch(self):
        if not self.examples:
            self.create_training_dataset()
        if self.index + self.traj_batch_size > len(self.examples):
            self.index = 0
            if not self.sample_live:
                random.shuffle(self.examples)
            else:
                self.create_training_dataset()
        batch = self.examples[self.index:self.index + self.traj_batch_size]
        return batch

    def _compute_loss(self, batch):
        xp = self.trex_network.xp
        preprocessed = {
            'i' : [batch_states([transition["obs"] for transition in example[0]], xp, self._phi)
                               for example in batch],
            'j' : [batch_states([transition["obs"] for transition in example[1]], xp, self._phi)
                                           for example in batch],
            'label' : xp.array([example[2] for example in batch])
        }
        rewards_i = [F.sum(self.trex_network(preprocessed['i'][i])) for i in range(len(preprocessed['i']))]
        rewards_j = [F.sum(self.trex_network(preprocessed['j'][i])) for i in range(len(preprocessed['j']))]
        rewards_i = F.expand_dims(F.stack(rewards_i), 1)
        rewards_j = F.expand_dims(F.stack(rewards_j), 1)
        predictions = F.concat((rewards_i, rewards_j))
        mean_loss = F.mean(F.softmax_cross_entropy(predictions,
                                                   preprocessed['label']))
        return mean_loss

    def _train(self):
        for _ in range(self.steps):
            print("Performed update...")
            # get batch of traj pairs
            batch = self.get_training_batch()
            # do updates
            loss = self._compute_loss(batch)
            self.trex_network.cleargrads()
            loss.backward()
            self.opt.update()
        if self.save_network:
            pass

        # at the end save the network

    def _phi(self, x):
        return np.asarray(x, dtype=np.float32) / 255
