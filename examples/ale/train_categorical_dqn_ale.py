from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

import gym
gym.undo_logger_setup()  # NOQA
import chainer
import numpy as np

import chainerrl
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer

import atari_wrappers


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
    parser.add_argument('--use-sdl', action='store_true', default=False)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--final-epsilon', type=float, default=0.1)
    parser.add_argument('--eval-epsilon', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--max-episode-len', type=int,
                        default=5 * 60 * 60 // 4,  # 5 minutes with 60/4 fps
                        help='Maximum number of steps for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',
                        type=int, default=10 ** 4)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
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
            atari_wrappers.make_atari(args.env),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            misc.env_modifiers.make_rendered(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n

    n_atoms = 51
    v_max = 10
    v_min = -10
    q_func = chainerrl.links.Sequence(
        chainerrl.links.NatureDQNHead(),
        chainerrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(
            None, n_actions, n_atoms, v_min, v_max,
            n_hidden_channels=0, n_hidden_layers=0),
    )

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as https://arxiv.org/abs/1707.06887
    opt = chainer.optimizers.Adam(2.5e-4, eps=1e-2 / args.batch_size)
    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = chainerrl.agents.CategoricalDQN(
        q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
        explorer=explorer, replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator='mean',
        phi=phi,
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
        # In testing DQN, randomly select 5% of actions
        eval_explorer = explorers.ConstantEpsilonGreedy(
            args.eval_epsilon, lambda: np.random.randint(n_actions))
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_explorer=eval_explorer,
            save_best_so_far_agent=False,
            max_episode_len=args.max_episode_len,
            eval_env=eval_env,
        )


if __name__ == '__main__':
    main()
