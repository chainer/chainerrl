from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import multiprocessing as mp
import os
import tempfile
import unittest

from chainer import testing
import mock

import chainerrl
from chainerrl.experiments.train_agent_async import train_loop


@testing.parameterize(*testing.product({
    'num_envs': [1, 2],
    'max_episode_len': [None, 2],
}))
class TestTrainAgentAsync(unittest.TestCase):

    def test(self):

        steps = 50

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

        # Mock states cannot be shared among processes. To check states of mock
        # objects, threading is used instead of multiprocessing.
        # Because threading.Thread does not have .exitcode attribute, we
        # add the attribute manually to avoid an exception.
        import threading

        # Mock.call_args_list does not seem thread-safe
        hook_lock = threading.Lock()
        hook = mock.Mock()

        def hook_locked(*args, **kwargs):
            with hook_lock:
                return hook(*args, **kwargs)

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
                global_step_hooks=[hook_locked],
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
            if self.num_envs == 1:
                # If num_envs == 1, a hook should be called sequentially.
                # step starts with 1
                self.assertEqual(args[2], i + 1)
        if self.num_envs > 1:
            # If num_envs > 1, a hook may not be called sequentially.
            # Thus, we only check if they are called for each step.
            hook_steps = [call[0][2] for call in hook.call_args_list]
            self.assertEqual(
                list(range(1, len(hook.call_args_list) + 1)),
                sorted(hook_steps),
            )

        # Agent should be saved
        agent.save.assert_called_once_with(
            os.path.join(outdir, '{}_finish'.format(steps)))


class TestTrainLoop(unittest.TestCase):

    def test_needs_reset(self):

        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
        env = mock.Mock()
        # First episode: 0 -> 1 -> 2 -> 3 (reset)
        # Second episode: 4 -> 5 -> 6 -> 7 (done)
        env.reset.side_effect = [('state', 0), ('state', 4)]
        env.step.side_effect = [
            (('state', 1), 0, False, {}),
            (('state', 2), 0, False, {}),
            (('state', 3), 0, False, {'needs_reset': True}),
            (('state', 5), -0.5, False, {}),
            (('state', 6), 0, False, {}),
            (('state', 7), 1, True, {}),
        ]

        counter = mp.Value('i', 0)
        episodes_counter = mp.Value('i', 0)
        training_done = mp.Value('b', False)  # bool
        train_loop(
            process_idx=0,
            env=env,
            agent=agent,
            steps=5,
            outdir=outdir,
            counter=counter,
            episodes_counter=episodes_counter,
            training_done=training_done,
        )

        self.assertEqual(agent.act_and_train.call_count, 5)
        self.assertEqual(agent.stop_episode_and_train.call_count, 2)

        self.assertEqual(env.reset.call_count, 2)
        self.assertEqual(env.step.call_count, 5)
