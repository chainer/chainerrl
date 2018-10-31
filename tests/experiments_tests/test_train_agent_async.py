from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import tempfile
import unittest

from chainer import testing
import mock

import chainerrl


@testing.parameterize(*testing.product({
    'num_envs': [1, 2],
    'max_episode_len': [None, 2],
}))
class TestTrainAgentAsync(unittest.TestCase):

    def test(self):

        steps = 5

        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
        agent.shared_attributes = []

        def _make_env(process_idx, test):
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

        # Keep references to mock envs to check their states later
        envs = [_make_env(i, test=False) for i in range(self.num_envs)]
        eval_envs = [_make_env(i, test=True) for i in range(self.num_envs)]

        def make_env(process_idx, test):
            if test:
                return eval_envs[process_idx]
            else:
                return envs[process_idx]

        hook = mock.Mock()

        # Mock states cannot be shared among processes. To check states of mock
        # objects, threading is used instead of multiprocessing.
        # Because threading.Thread does not have .exitcode attribute, we
        # add the attribute manually to avoid an exception.
        import threading
        with mock.patch('multiprocessing.Process', threading.Thread),\
            mock.patch.object(
                threading.Thread, 'exitcode', create=True, new=0):
            chainerrl.experiments.train_agent_async(
                processes=self.num_envs,
                agent=agent,
                make_env=make_env,
                steps=steps,
                outdir=outdir,
                max_episode_len=self.max_episode_len,
                global_step_hooks=[hook],
            )

        if self.num_envs == 1:
            self.assertEqual(agent.act_and_train.call_count, steps)
        elif self.num_envs > 1:
            self.assertGreater(agent.act_and_train.call_count, steps)

        # All the envs (including eval envs) should to be closed
        for env in envs + eval_envs:
            env.close.assert_called_once()

        if self.num_envs == 1:
            self.assertEqual(hook.call_count, steps)
        elif self.num_envs > 1:
            self.assertGreater(hook.call_count, steps)

        # A hook receives (env, agent, step)
        for i, call in enumerate(hook.call_args_list):
            args, kwargs = call
            self.assertTrue(any(args[0] == env for env in envs))
            self.assertEqual(args[1], agent)
            # step starts with 1
            self.assertEqual(args[2], i + 1)
