from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import unittest


from chainer import links as L
from chainer import optimizers
from chainer import testing
import numpy as np

from chainerrl import agents
from chainerrl import explorers
from chainerrl import links
from chainerrl.action_value import DiscreteActionValue
from chainerrl.misc import download_model
from chainerrl.misc import pretrained_models
from chainerrl import replay_buffer


@testing.parameterize(*testing.product(
    {
        'pretrained_type': ["best", "final"],
    }
))
class TestLoadDQN(unittest.TestCase):

    def _test_load_dqn(self, gpu):
        q_func = links.Sequence(
                    links.NatureDQNHead(),
                    L.Linear(512, 4),
                    DiscreteActionValue)

        opt = optimizers.RMSpropGraves(
            lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)
        opt.setup(q_func)

        rbuf = replay_buffer.ReplayBuffer(10 ** 6)

        explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0, end_epsilon=0.1,
            decay_steps=10 ** 6,
            random_action_func=lambda: np.random.randint(4))

        agent = agents.DQN(q_func, opt, rbuf, gpu=gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=4000,
                  target_update_interval=10 ** 4,
                  clip_delta=True,
                  update_interval=4,
                  batch_accumulator='sum',
                  phi=lambda x : x)

        model, exists = download_model("DQN", "BreakoutNoFrameskip-v4",
                               model_type=self.pretrained_type)
        agent.load(model)
        # TODO: have agent act?
        assert exists

    def test_cpu(self):
        self._test_load_dqn(gpu=None)

    @testing.attr.gpu
    def test_gpu(self):
        self._test_load_dqn(gpu=0)
