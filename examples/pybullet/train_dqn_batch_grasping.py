from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import functools
import os

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import gym
import gym.spaces
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer


class CastAction(gym.ActionWrapper):
    """Cast actions to a given type."""

    def __init__(self, env, type_):
        super().__init__(env)
        self.type_ = type_

    def _action(self, action):
        return self.type_(action)


class TransposeObservation(gym.ObservationWrapper):
    """Transpose observations."""

    def __init__(self, env, axes):
        super().__init__(env)
        self._axes = axes

    def _observation(self, observation):
        return observation.transpose(*self._axes)


class ObserveElapsedSteps(gym.Wrapper):
    """Observe the number of elapsed steps in an episode."""

    def __init__(self, env, max_steps):
        super().__init__(env)
        self._max_steps = max_steps
        self._elapsed_steps = 0
        self.observation_space = gym.spaces.Tuple((
            env.observation_space,
            gym.spaces.Discrete(self._max_steps + 1),
        ))

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset(), self._elapsed_steps

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        assert self._elapsed_steps <= self._max_steps
        return (observation, self._elapsed_steps), reward, done, info


class SingleSharedBias(chainer.Chain):
    """Single shared bias used in the Double DQN paper.

    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.

    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.bias = chainer.Parameter(0, shape=1)

    def __call__(self, x):
        return x + F.broadcast_to(self.bias, x.shape)


class GraspingQFunction(chainer.Chain):

    def __init__(self, n_actions, max_episode_steps):
        super().__init__()
        with self.init_scope():
            self.embed = L.EmbedID(max_episode_steps + 1, 3136)
            self.image2hidden = chainer.Sequential(
                L.Convolution2D(None, 32, 8, stride=4),
                F.relu,
                L.Convolution2D(None, 64, 4, stride=2),
                F.relu,
                L.Convolution2D(None, 64, 3, stride=1),
                functools.partial(F.reshape, shape=(-1, 3136)),
            )
            self.hidden2out = chainer.Sequential(
                L.Linear(None, 512),
                F.relu,
                L.Linear(None, n_actions, nobias=True),
                SingleSharedBias(),
                DiscreteActionValue,
            )

    def forward(self, x):
        image, steps = x
        h = self.image2hidden(image) * F.sigmoid(self.embed(steps))
        return self.hidden2out(h)


def parse_agent(agent):
    return {'DQN': agents.DQN,
            'DoubleDQN': agents.DoubleDQN,
            'PAL': agents.PAL}[agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--final-epsilon', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',
                        type=int, default=3 * 10 ** 4)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--lr', type=float, default=6.25e-5,
                        help='Learning rate')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of envs run in parallel.')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    max_episode_steps = 8

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv  # NOQA
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        # Set a random seed for this subprocess
        misc.set_random_seed(env_seed)
        env = KukaDiverseObjectEnv(
            isDiscrete=True,
            renders=args.render and (args.demo or not test),
            height=84,
            width=84,
            maxSteps=max_episode_steps,
        )
        # (84, 84, 3) -> (3, 84, 84)
        env = TransposeObservation(env, (2, 0, 1))
        env = ObserveElapsedSteps(env, max_episode_steps)
        env = CastAction(env, int)
        env.seed(int(env_seed))
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test)
                for idx in range(args.num_envs)])

    eval_env = make_batch_env(test=False)
    n_actions = eval_env.action_space.n

    q_func = GraspingQFunction(n_actions, max_episode_steps)

    # Draw the computational graph and save it in the output directory.
    fake_obs = (
        np.zeros((3, 84, 84), dtype=np.float32)[None],
        np.zeros(1, dtype=np.int32)[None],
    )
    chainerrl.misc.draw_computational_graph(
        [q_func(fake_obs)],
        os.path.join(args.outdir, 'model'))

    # Use the hyper parameters of the Nature paper
    opt = optimizers.RMSpropGraves(
        lr=args.lr, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    # Anneal beta from beta0 to 1 throughout training
    betasteps = args.steps / args.update_interval
    rbuf = replay_buffer.PrioritizedReplayBuffer(
        10 ** 6, alpha=0.6, beta0=0.4, betasteps=betasteps)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        image, elapsed_steps = x
        # Normalize RGB values: [0, 255] -> [0, 1]
        norm_image = np.asarray(image, dtype=np.float32) / 255
        return norm_image, elapsed_steps

    def batch_states(obss, xp, phi):
        return chainer.dataset.concat_examples(
            [phi(obs) for obs in obss], device=args.gpu)

    agent = chainerrl.agents.DoubleDQN(
        q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
        explorer=explorer, replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator='sum',
        phi=phi,
        batch_states=batch_states,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=eval_env,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            log_interval=1000,
        )


if __name__ == '__main__':
    main()
