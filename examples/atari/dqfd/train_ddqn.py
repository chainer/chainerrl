"""Prioritized Dueling Double n-step DQN on ATARI with log rewards and
no clipping
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
from builtins import *  # NOQA

import chainerrl

from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer
from chainerrl.q_functions import DuelingDQN
from chainerrl.wrappers import atari_wrappers

from chainer import optimizers

from future import standard_library
standard_library.install_aliases()  # NOQA

import gym

import numpy as np

import json


class LogScaleReward(gym.RewardWrapper):
    """Convert reward to log scale as described in the DQfD paper
    https://arxiv.org/pdf/1704.03732.pdf

    Args:
        env: Env to wrap.
        scale (float): Scale factor.
    Attributes:
        scale: Scale factor.
        original_reward: Reward before casting.
    """

    def __init__(self, env):
        super().__init__(env)
        self.original_reward = None

    def reward(self, reward):
        self.original_reward = reward
        return np.sign(reward) * np.log(1 + np.abs(reward))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HeroNoFrameskip-v4',
                        help='OpenAI Atari domain to perform algorithm on.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--eval-n-steps', type=int, default=125000)
    parser.add_argument('--eval-interval', type=int, default=250000)
    parser.add_argument('--n-best-episodes', type=int, default=30)
    parser.add_argument('--target-update-interval',
                        type=int, default=10 ** 4,
                        help='Frequency (in timesteps) at which ' +
                        'the target network is updated.')
    parser.add_argument('--update-interval', type=int, default=4,
                        help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--replay-buffer-size', type=int, default=10**6)
    parser.add_argument('--num_step_return', type=int, default=10)
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=False)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, 0.001)
        else:
            # Log scale train environment
            env = LogScaleReward(env)
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    q_func = DuelingDQN(n_actions)

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # Use the same hyperparameters as the Nature paper
    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    betasteps = args.steps / args.update_interval
    rbuff = replay_buffer.PrioritizedReplayBuffer(
        args.replay_buffer_size, alpha=0.6,
        beta0=0.4, betasteps=betasteps,
        num_steps=args.num_step_return)

    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.01,
        decay_steps=10 ** 6,
        random_action_func=lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = agents.DoubleDQN(q_func, opt, rbuff, gpu=args.gpu, gamma=0.99,
                             explorer=explorer,
                             replay_start_size=args.replay_start_size,
                             target_update_interval=args.target_update_interval,
                             clip_delta=args.clip_delta,
                             update_interval=args.update_interval,
                             batch_accumulator='sum',
                             phi=phi)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=args.eval_n_steps,
            n_episodes=None)
        print('n_episodes: {} mean: {} median: {} stdev {}'.format(
            eval_stats['episodes'],
            eval_stats['mean'],
            eval_stats['median'],
            eval_stats['stdev']))
    else:
        logger = logging.getLogger(__name__)
        evaluator = experiments.Evaluator(agent=agent,
                                          n_steps=args.eval_n_steps,
                                          n_episodes=None,
                                          eval_interval=args.eval_interval,
                                          outdir=args.outdir,
                                          max_episode_len=None,
                                          env=eval_env,
                                          step_offset=0,
                                          save_best_so_far_agent=True,
                                          logger=logger)

        # Evaluate the agent BEFORE training begins
        evaluator.evaluate_and_update_max_score(t=0, episodes=0)

        experiments.train_agent(agent=agent,
                                env=env,
                                steps=args.steps,
                                outdir=args.outdir,
                                max_episode_len=None,
                                step_offset=0,
                                evaluator=evaluator,
                                successful_score=None,
                                step_hooks=[])

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        # run 30 evaluation episodes, each capped at 30 mins of play
        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=27000,
            logger=None)
        with open(os.path.join(args.outdir, 'bestscores.json'), 'w') as f:
            # temporary hack to handle python 2/3 support issues.
            # json dumps does not support non-string literal dict keys
            json_stats = json.dumps(stats)
            print(str(json_stats), file=f)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))


if __name__ == '__main__':
    main()
