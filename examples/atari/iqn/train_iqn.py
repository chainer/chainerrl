from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import functools
import json
import os

import chainer
import chainer.functions as F
import chainer.links as L
import gym
import numpy as np

import chainerrl
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer
from chainerrl.wrappers import atari_wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
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
    parser.add_argument('--final-epsilon', type=float, default=0.01)
    parser.add_argument('--eval-epsilon', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7)
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',
                        type=int, default=10 ** 4)
    parser.add_argument('--eval-interval', type=int, default=250000)
    parser.add_argument('--eval-n-steps', type=int, default=125000)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--batch-accumulator', type=str, default='mean',
                        choices=['mean', 'sum'])
    parser.add_argument('--quantile-thresholds-N', type=int, default=64)
    parser.add_argument('--quantile-thresholds-N-prime', type=int, default=64)
    parser.add_argument('--quantile-thresholds-K', type=int, default=32)
    parser.add_argument('--n-best-episodes', type=int, default=200)
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
            clip_rewards=not test)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, args.eval_epsilon)
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

    q_func = chainerrl.agents.iqn.ImplicitQuantileQFunction(
        psi=chainerrl.links.Sequence(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            F.relu,
            functools.partial(F.reshape, shape=(-1, 3136)),
        ),
        phi=chainerrl.links.Sequence(
            chainerrl.agents.iqn.CosineBasisLinear(64, 3136),
            F.relu,
        ),
        f=chainerrl.links.Sequence(
            L.Linear(None, 512),
            F.relu,
            L.Linear(None, n_actions),
        ),
    )

    # Draw the computational graph and save it in the output directory.
    fake_obss = np.zeros((4, 84, 84), dtype=np.float32)[None]
    fake_taus = np.zeros(32, dtype=np.float32)[None]
    chainerrl.misc.draw_computational_graph(
        [q_func(fake_obss)(fake_taus)],
        os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as https://arxiv.org/abs/1710.10044
    opt = chainer.optimizers.Adam(5e-5, eps=1e-2 / args.batch_size)
    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = chainerrl.agents.IQN(
        q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
        explorer=explorer, replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator=args.batch_accumulator,
        phi=phi,
        quantile_thresholds_N=args.quantile_thresholds_N,
        quantile_thresholds_N_prime=args.quantile_thresholds_N_prime,
        quantile_thresholds_K=args.quantile_thresholds_K,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=args.eval_n_steps,
            n_episodes=None,
        )
        print('n_steps: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_steps, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
        )

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        # run 200 evaluation episodes, each capped at 30 mins of play
        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=args.max_frames / 4,
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
