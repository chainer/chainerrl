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
from chainerrl.experiments import evaluator


@testing.parameterize(
    *testing.product({
        'save_best_so_far_agent': [True, False],
        'n_steps': [None, 1, 2],
        'n_episodes': [None, 1, 2],
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

        either_none = (self.n_steps is None) != (self.n_episodes is None)
        if not either_none:
            with self.assertRaises(AssertionError):
                agent_evaluator = evaluator.Evaluator(
                    agent=agent,
                    env=env,
                    n_steps=self.n_steps,
                    n_episodes=self.n_episodes,
                    eval_interval=3,
                    outdir=outdir,
                    max_episode_len=None,
                    step_offset=0,
                    save_best_so_far_agent=self.save_best_so_far_agent,
                )
        else:
            value = self.n_steps or self.n_episodes
            agent_evaluator = evaluator.Evaluator(
                agent=agent,
                env=env,
                n_steps=self.n_steps,
                n_episodes=self.n_episodes,
                eval_interval=3,
                outdir=outdir,
                max_episode_len=None,
                step_offset=0,
                save_best_so_far_agent=self.save_best_so_far_agent,
            )

            agent_evaluator.evaluate_if_necessary(t=1, episodes=1)
            self.assertEqual(agent.act.call_count, 0)

            agent_evaluator.evaluate_if_necessary(t=2, episodes=2)
            self.assertEqual(agent.act.call_count, 0)

            # First evaluation
            agent_evaluator.evaluate_if_necessary(t=3, episodes=3)
            self.assertEqual(agent.act.call_count, value)
            self.assertEqual(agent.stop_episode.call_count, value)
            if self.save_best_so_far_agent:
                self.assertEqual(agent.save.call_count, 1)
            else:
                self.assertEqual(agent.save.call_count, 0)

            # Second evaluation with the same score
            agent_evaluator.evaluate_if_necessary(t=6, episodes=6)
            self.assertEqual(agent.act.call_count, 2 * value)
            self.assertEqual(agent.stop_episode.call_count, 2 * value)
            if self.save_best_so_far_agent:
                self.assertEqual(agent.save.call_count, 1)
            else:
                self.assertEqual(agent.save.call_count, 0)

            # Third evaluation with a better score
            env.step.return_value = ('obs', 1, True, {})
            agent_evaluator.evaluate_if_necessary(t=9, episodes=9)
            self.assertEqual(agent.act.call_count, 3 * value)
            self.assertEqual(agent.stop_episode.call_count, 3 * value)
            if self.save_best_so_far_agent:
                self.assertEqual(agent.save.call_count, 2)
            else:
                self.assertEqual(agent.save.call_count, 0)


@testing.parameterize(
    *testing.product({
        'save_best_so_far_agent': [True, False],
        'n_episodes': [1, 2],
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

        agent_evaluator = evaluator.AsyncEvaluator(
            n_steps=None,
            n_episodes=self.n_episodes,
            eval_interval=3,
            outdir=outdir,
            max_episode_len=None,
            step_offset=0,
            save_best_so_far_agent=self.save_best_so_far_agent,
        )

        agent_evaluator.evaluate_if_necessary(
            t=1, episodes=1, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, 0)

        agent_evaluator.evaluate_if_necessary(
            t=2, episodes=2, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, 0)

        # First evaluation
        agent_evaluator.evaluate_if_necessary(
            t=3, episodes=3, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, self.n_episodes)
        self.assertEqual(agent.stop_episode.call_count, self.n_episodes)
        if self.save_best_so_far_agent:
            self.assertEqual(agent.save.call_count, 1)
        else:
            self.assertEqual(agent.save.call_count, 0)

        # Second evaluation with the same score
        agent_evaluator.evaluate_if_necessary(
            t=6, episodes=6, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, 2 * self.n_episodes)
        self.assertEqual(agent.stop_episode.call_count, 2 * self.n_episodes)
        if self.save_best_so_far_agent:
            self.assertEqual(agent.save.call_count, 1)
        else:
            self.assertEqual(agent.save.call_count, 0)

        # Third evaluation with a better score
        env.step.return_value = ('obs', 1, True, {})
        agent_evaluator.evaluate_if_necessary(
            t=9, episodes=9, env=env, agent=agent)
        self.assertEqual(agent.act.call_count, 3 * self.n_episodes)
        self.assertEqual(agent.stop_episode.call_count, 3 * self.n_episodes)
        if self.save_best_so_far_agent:
            self.assertEqual(agent.save.call_count, 2)
        else:
            self.assertEqual(agent.save.call_count, 0)


@testing.parameterize(
    *testing.product({
        'n_episodes': [None, 1],
        'n_timesteps': [2, 5, 6],
    })
)
class TestRunTimeBasedEvaluationEpisode(unittest.TestCase):

    def test_timesteps(self):
        agent = mock.Mock()
        env = mock.Mock()
        # First episode: 0 -> 1 -> 2 -> 3 (reset)
        # Second episode: 4 -> 5 -> 6 -> 7 (done)
        env.reset.side_effect = [('state', 0), ('state', 4)]
        env.step.side_effect = [
            (('state', 1), 0.1, False, {}),
            (('state', 2), 0.2, False, {}),
            (('state', 3), 0.3, False, {'needs_reset': True}),
            (('state', 5), -0.5, False, {}),
            (('state', 6), 0, False, {}),
            (('state', 7), 1, True, {}),
        ]

        if self.n_episodes:
            with self.assertRaises(AssertionError):
                scores = evaluator.run_evaluation_episodes(
                    env, agent,
                    n_steps=self.n_timesteps,
                    n_episodes=self.n_episodes)
        else:
            scores = evaluator.run_evaluation_episodes(
                env, agent,
                n_steps=self.n_timesteps,
                n_episodes=self.n_episodes)
            if self.n_timesteps == 2:
                self.assertAlmostEqual(len(scores), 1)
                self.assertAlmostEqual(scores[0], 0.3)
                self.assertEqual(agent.stop_episode.call_count, 1)
            elif self.n_timesteps == 5:
                self.assertAlmostEqual(len(scores), 1)
                self.assertAlmostEqual(scores[0], 0.6)
                self.assertEqual(agent.stop_episode.call_count, 2)
            else:
                self.assertAlmostEqual(len(scores), 2)
                self.assertAlmostEqual(scores[0], 0.6)
                self.assertAlmostEqual(scores[1], 0.5)
                self.assertEqual(agent.stop_episode.call_count, 2)


class TestRunEvaluationEpisode(unittest.TestCase):

    def test_needs_reset(self):
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
        scores = evaluator.run_evaluation_episodes(
            env, agent, n_steps=None, n_episodes=2)
        self.assertAlmostEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 0)
        self.assertAlmostEqual(scores[1], 0.5)
        self.assertAlmostEqual(agent.stop_episode.call_count, 2)


@testing.parameterize(
    *testing.product({
        'n_episodes': [None, 1],
        'n_timesteps': [2, 5, 6],
    })
)
class TestBatchRunTimeBasedEvaluationEpisode(unittest.TestCase):

    def test_timesteps(self):
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
                    (('state', 2), 0.1, False, {}),
                    (('state', 3), 0.2, False, {'needs_reset': True}),
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
                    (('state', 5), -0.6, False, {}),
                    (('state', 6), 0, False, {}),
                    (('state', 7), 1, True, {}),
                ]
            return env

        vec_env = chainerrl.envs.SerialVectorEnv(
            [make_env(i) for i in range(2)])
        if self.n_episodes:
            with self.assertRaises(AssertionError):
                scores = evaluator.batch_run_evaluation_episodes(
                    vec_env, agent,
                    n_steps=self.n_timesteps,
                    n_episodes=self.n_episodes)
        else:
            # First Env:  [1   2   (3_a)  5  6   (7_a)]
            # Second Env: [(1)(3_b) 5     6 (7_b)]
            scores = evaluator.batch_run_evaluation_episodes(
                vec_env, agent,
                n_steps=self.n_timesteps,
                n_episodes=self.n_episodes)
            if self.n_timesteps == 2:
                self.assertAlmostEqual(len(scores), 1)
                self.assertAlmostEqual(scores[0], 0.1)
                self.assertEqual(agent.batch_observe.call_count, 2)
            else:
                self.assertAlmostEqual(len(scores), 3)
                self.assertAlmostEqual(scores[0], 0.3)
                self.assertAlmostEqual(scores[1], 2.0)
                self.assertAlmostEqual(scores[2], 3.0)
            # batch_reset should be all True
            self.assertTrue(all(agent.batch_observe.call_args[0][3]))


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
                    (('state', 5), -0.6, False, {}),
                    (('state', 6), 0, False, {}),
                    (('state', 7), 1, True, {}),
                ]
            return env

        vec_env = chainerrl.envs.SerialVectorEnv(
            [make_env(i) for i in range(2)])

        # First Env: [1 2 (3_a) 5 6 (7_a)]
        # Second Env: [(1) (3_b) 5 6 (7_b)]
        # Results: (1), (3a), (3b), (7b)
        scores = evaluator.batch_run_evaluation_episodes(
            vec_env, agent, n_steps=None, n_episodes=4)
        self.assertAlmostEqual(len(scores), 4)
        self.assertAlmostEqual(scores[0], 0)
        self.assertAlmostEqual(scores[1], 2)
        self.assertAlmostEqual(scores[2], 3)
        self.assertAlmostEqual(scores[3], 0.4)
        # batch_reset should be all True
        self.assertTrue(all(agent.batch_observe.call_args[0][3]))
