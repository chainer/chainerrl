import os
import tempfile
import unittest
from unittest import mock

import chainer
from chainer import testing

import chainerrl.experiments as experiments


@testing.parameterize(
    *testing.product({
        'n_steps': [None, 5],
        'n_episodes': [None, 1],
    })
)
@testing.with_requires('chainer>=5.0.0')
class TestCollectDemos(unittest.TestCase):
    def test(self):

        outdir = tempfile.mkdtemp()

        env = mock.Mock()
        # Reaches the terminal state after five actions
        env.reset.side_effect = [('state', 0)]
        env.step.side_effect = [
            (('state', 1), 0, False, {}),
            (('state', 2), 0, False, {}),
            (('state', 3), -0.5, False, {}),
            (('state', 4), 0, False, {}),
            (('state', 5), 1, True, {}),
        ]

        agent = mock.Mock()
        agent.act.side_effect = [0, 1, 2, 3, 4]
        if (not self.n_steps and not self.n_episodes) or \
                (self.n_steps and self.n_episodes):
            with self.assertRaises(AssertionError):
                experiments.collect_demonstrations(agent,
                                                   env,
                                                   self.n_steps,
                                                   self.n_episodes,
                                                   outdir,
                                                   None,
                                                   None)
            return
        experiments.collect_demonstrations(agent,
                                           env,
                                           self.n_steps,
                                           self.n_episodes,
                                           outdir,
                                           max_episode_len=None,
                                           logger=None)

        self.assertEqual(agent.act.call_count, 5)
        self.assertEqual(agent.stop_episode.call_count, 1)

        self.assertEqual(env.reset.call_count, 1)
        self.assertEqual(env.step.call_count, 5)

        true_states = [0, 1, 2, 3, 4]
        true_next_states = [1, 2, 3, 4, 5]
        true_actions = [0, 1, 2, 3, 4]
        true_rewards = [0, 0, -0.5, 0, 1]
        with chainer.datasets.open_pickle_dataset(
                os.path.join(outdir, "demos.pickle")) as dataset:
            self.assertEqual(len(dataset), 5)
            for i in range(5):
                obs, a, r, new_obs, _, _ = dataset[i]
                self.assertEqual(obs[1], true_states[i])
                self.assertEqual(a, true_actions[i])
                self.assertEqual(r, true_rewards[i])
                self.assertEqual(new_obs[1], true_next_states[i])

    def test_needs_reset(self):

        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
        agent.act.side_effect = [0, 1, 2, 3, 4]
        env = mock.Mock()
        # First episode: 0 -> 1 -> 2 -> 3 (reset)
        # Second episode: 4 -> 5 -> 6 -> 7 (done)
        env.reset.side_effect = [('state', 0), ('state', 4)]
        env.step.side_effect = [
            (('state', 1), 0, False, {}),
            (('state', 2), 0, False, {}),
            (('state', 3), 0, False, {'needs_reset': True}),
            (('state', 5), -0.5, False, {}),
            (('state', 7), 1, True, {}),
        ]
        if (not self.n_steps and not self.n_episodes) or \
                (self.n_steps and self.n_episodes):
            with self.assertRaises(AssertionError):
                experiments.collect_demonstrations(agent,
                                                   env,
                                                   self.n_steps,
                                                   self.n_episodes,
                                                   outdir,
                                                   None,
                                                   None)
            return

        steps = self.n_steps
        # 2 to match the mock env, b/c test is parameterized by episodes=1
        episodes = 2 if self.n_episodes else self.n_episodes
        experiments.collect_demonstrations(
            agent,
            env,
            steps,
            episodes,
            outdir,
            max_episode_len=None,
            logger=None)
        self.assertEqual(agent.act.call_count, 5)
        self.assertEqual(agent.stop_episode.call_count, 2)
        self.assertEqual(env.reset.call_count, 2)
        self.assertEqual(env.step.call_count, 5)

        true_states = [0, 1, 2, 4, 5]
        true_next_states = [1, 2, 3, 5, 7]
        true_actions = [0, 1, 2, 3, 4]
        true_rewards = [0, 0, 0, -0.5, 1]
        with chainer.datasets.open_pickle_dataset(
                os.path.join(outdir, "demos.pickle")) as dataset:
            self.assertEqual(len(dataset), 5)
            for i in range(5):
                obs, a, r, new_obs, _, _ = dataset[i]
                self.assertEqual(obs[1], true_states[i])
                self.assertEqual(a, true_actions[i])
                self.assertEqual(r, true_rewards[i])
                self.assertEqual(new_obs[1], true_next_states[i])
