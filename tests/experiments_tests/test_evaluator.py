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


@testing.parameterize(
    *testing.product({
        'save_best_so_far_agent': [True, False],
        'n_runs': [1, 2],
    })
)
class TestEvaluator(unittest.TestCase):

    def test_evaluate_if_necessary(self):
        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
        agent.act.return_value = 'action'
        agent.get_statistics.return_value = []

        env = mock.Mock()
        env.reset.return_value = 'obs'
        env.step.return_value = ('obs', 0, True, {})

        evaluator = chainerrl.experiments.evaluator.Evaluator(
            agent=agent,
            env=env,
            n_runs=self.n_runs,
            eval_interval=3,
            outdir=outdir,
            max_episode_len=None,
            explorer=None,
            step_offset=0,
            save_best_so_far_agent=self.save_best_so_far_agent,
        )

        evaluator.evaluate_if_necessary(t=1, episodes=1)
        self.assertEqual(agent.act.call_count, 0)

        evaluator.evaluate_if_necessary(t=2, episodes=2)
        self.assertEqual(agent.act.call_count, 0)

        # First evaluation
        evaluator.evaluate_if_necessary(t=3, episodes=3)
        self.assertEqual(agent.act.call_count, self.n_runs)
        self.assertEqual(agent.stop_episode.call_count, self.n_runs)
        if self.save_best_so_far_agent:
            self.assertEqual(agent.save.call_count, 1)
        else:
            self.assertEqual(agent.save.call_count, 0)

        # Second evaluation with the same score
        evaluator.evaluate_if_necessary(t=6, episodes=6)
        self.assertEqual(agent.act.call_count, 2 * self.n_runs)
        self.assertEqual(agent.stop_episode.call_count, 2 * self.n_runs)
        if self.save_best_so_far_agent:
            self.assertEqual(agent.save.call_count, 1)
        else:
            self.assertEqual(agent.save.call_count, 0)

        # Third evaluation with a better score
        env.step.return_value = ('obs', 1, True, {})
        evaluator.evaluate_if_necessary(t=9, episodes=9)
        self.assertEqual(agent.act.call_count, 3 * self.n_runs)
        self.assertEqual(agent.stop_episode.call_count, 3 * self.n_runs)
        if self.save_best_so_far_agent:
            self.assertEqual(agent.save.call_count, 2)
        else:
            self.assertEqual(agent.save.call_count, 0)


@testing.parameterize(
    *testing.product({
        'save_best_so_far_agent': [True, False],
        'n_runs': [1, 2],
    })
)
class TestAsyncEvaluator(unittest.TestCase):

    def test_evaluate_if_necessary(self):
        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
        agent.act.return_value = 'action'
        agent.get_statistics.return_value = []

        env = mock.Mock()
        env.reset.return_value = 'obs'
        env.step.return_value = ('obs', 0, True, {})

        evaluator = chainerrl.experiments.evaluator.AsyncEvaluator(
            n_runs=self.n_runs,
            eval_interval=3,
            outdir=outdir,
            max_episode_len=None,
            explorer=None,
            step_offset=0,
            save_best_so_far_agent=self.save_best_so_far_agent,
        )

        evaluator.evaluate_if_necessary(t=1, episodes=1, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, 0)

        evaluator.evaluate_if_necessary(t=2, episodes=2, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, 0)

        # First evaluation
        evaluator.evaluate_if_necessary(t=3, episodes=3, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, self.n_runs)
        self.assertEqual(agent.stop_episode.call_count, self.n_runs)
        if self.save_best_so_far_agent:
            self.assertEqual(agent.save.call_count, 1)
        else:
            self.assertEqual(agent.save.call_count, 0)

        # Second evaluation with the same score
        evaluator.evaluate_if_necessary(t=6, episodes=6, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, 2 * self.n_runs)
        self.assertEqual(agent.stop_episode.call_count, 2 * self.n_runs)
        if self.save_best_so_far_agent:
            self.assertEqual(agent.save.call_count, 1)
        else:
            self.assertEqual(agent.save.call_count, 0)

        # Third evaluation with a better score
        env.step.return_value = ('obs', 1, True, {})
        evaluator.evaluate_if_necessary(t=9, episodes=9, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, 3 * self.n_runs)
        self.assertEqual(agent.stop_episode.call_count, 3 * self.n_runs)
        if self.save_best_so_far_agent:
            self.assertEqual(agent.save.call_count, 2)
        else:
            self.assertEqual(agent.save.call_count, 0)
