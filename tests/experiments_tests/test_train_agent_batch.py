from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import math
import tempfile
import unittest

from chainer import testing
import mock

import chainerrl


@testing.parameterize(*testing.product({
    'num_envs': [1, 2],
    'max_episode_len': [None, 2],
}))
class TestTrainAgentBatch(unittest.TestCase):

    def test(self):

        steps = 5

        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
        agent.batch_act_and_train.side_effect = [[1] * self.num_envs] * 1000

        def make_env():
            env = mock.Mock()
            env.reset.side_effect = [('state', 0)] * 1000
            if self.max_episode_len is None:
                # Episodic env that terminates after 5 actions
                env.step.side_effect = [
                    (('state', 1), 0, False, {}),
                    (('state', 2), 0, False, {}),
                    (('state', 3), -0.5, False, {}),
                    (('state', 4), 0, False, {}),
                    (('state', 5), 1, True, {}),
                ] * 1000
            else:
                # Continuing env
                env.step.side_effect = [
                    (('state', 1), 0, False, {}),
                ] * 1000
            return env

        vec_env = chainerrl.envs.SerialVectorEnv(
            [make_env() for _ in range(self.num_envs)])

        hook = mock.Mock()

        chainerrl.experiments.train_agent_batch(
            agent=agent,
            env=vec_env,
            steps=steps,
            outdir=outdir,
            max_episode_len=self.max_episode_len,
            step_hooks=[hook],
        )

        iters = math.ceil(steps / self.num_envs)
        self.assertEqual(agent.batch_act_and_train.call_count, iters)
        self.assertEqual(agent.batch_observe_and_train.call_count, iters)

        for env in vec_env.envs:
            if self.max_episode_len is None:
                if self.num_envs == 1:
                    # Both in the beginning and after termination
                    self.assertEqual(env.reset.call_count, 2)
                elif self.num_envs == 2:
                    # Only in the beginning
                    self.assertEqual(env.reset.call_count, 1)
                else:
                    assert False
            elif self.max_episode_len == 2:
                if self.num_envs == 1:
                    # In the beginning, after 2 and 4 iterations
                    self.assertEqual(env.reset.call_count, 3)
                elif self.num_envs == 2:
                    # In the beginning, after 2 iterations
                    self.assertEqual(env.reset.call_count, 2)
                else:
                    assert False
            self.assertEqual(env.step.call_count, iters)

        if steps % self.num_envs == 0:
            self.assertEqual(hook.call_count, steps)
        else:
            self.assertEqual(hook.call_count, self.num_envs * iters)

        # A hook receives (env, agent, step)
        for i, call in enumerate(hook.call_args_list):
            args, kwargs = call
            self.assertEqual(args[0], vec_env)
            self.assertEqual(args[1], agent)
            # step starts with 1
            self.assertEqual(args[2], i + 1)
