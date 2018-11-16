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


class TestRunEvaluationEpisode(unittest.TestCase):

    def test_needs_reset(self):
        agent = mock.Mock()
        env = mock.Mock()
        # Reaches the terminal state after five actions
        env.reset.side_effect = [('state', 0), ('state', 4)]
        env.step.side_effect = [
            (('state', 1), 0, False, {}),
            (('state', 2), 0, False, {}),
            (('state', 3), 0, False, {'needs_reset': True}),
            (('state', 5), -0.5, False, {}),
            (('state', 6), 0, False, {}),
            (('state', 7), 1, True, {}),
        ]
        scores = chainerrl.experiments.evaluator.run_evaluation_episodes(
            env, agent, n_runs=2)
        self.assertAlmostEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 0)
        self.assertAlmostEqual(scores[1], 0.5)


class TestBatchRunEvaluationEpisode(unittest.TestCase):

    def test_needs_reset(self):
        agent = mock.Mock()
        agent.batch_act.side_effect = [[1, 1]] * 5

        def make_env(idx):
            env = mock.Mock()
            if idx == 0:
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
            else:
                # First episode: 0 -> 1 (reset)
                # Second episode: 2 -> 3 (reset)
                # Third episode: 4 -> 5 -> 6 -> 7 (done)
                env.reset.side_effect = [
                    ('state', 0), ('state', 2), ('state', 4)]
                env.step.side_effect = [
                    (('state', 1), 2, False, {'needs_reset': True}),
                    (('state', 3), 3, False, {'needs_reset': True}),
                    (('state', 5), -0.5, False, {}),
                    (('state', 6), 0, False, {}),
                    (('state', 7), 1, True, {}),
                ]
            return env

        vec_env = chainerrl.envs.SerialVectorEnv(
            [make_env(i) for i in range(2)])

        scores = chainerrl.experiments.evaluator.batch_run_evaluation_episodes(
            vec_env, agent, n_runs=4)
        self.assertAlmostEqual(len(scores), 4)
        self.assertAlmostEqual(scores[0], 2)
        self.assertAlmostEqual(scores[1], 3)
        self.assertAlmostEqual(scores[2], 0)
        self.assertAlmostEqual(scores[3], 0.5)
