from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse
import os

import chainer
from chainer import links as L

from chainerrl.action_value import DiscreteActionValue
from chainerrl.agents import acer
from chainerrl.distribution import SoftmaxDistribution
from chainerrl.envs import ale
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl.replay_buffer import EpisodicReplayBuffer

from dqn_phi import dqn_phi


def main():

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--use-sdl', action='store_true')
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--replay-start-size', type=int, default=10000)
    parser.add_argument('--n-times-replay', type=int, default=4)
    parser.add_argument('--max-episode-len', type=int, default=10000)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.set_defaults(use_sdl=False)
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    if args.use_lstm:
        model = acer.ACERSharedModel(
            shared=links.Sequence(
                links.NIPSDQNHead(),
                L.LSTM(256, 256)),
            pi=links.Sequence(
                L.Linear(256, n_actions),
                SoftmaxDistribution),
            q=links.Sequence(
                L.Linear(256, n_actions),
                DiscreteActionValue),
        )
    else:
        model = acer.ACERSharedModel(
            shared=links.NIPSDQNHead(),
            pi=links.Sequence(
                L.Linear(256, n_actions),
                SoftmaxDistribution),
            q=links.Sequence(
                L.Linear(256, n_actions),
                DiscreteActionValue),
        )
    opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=4e-3, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))
    replay_buffer = EpisodicReplayBuffer(10 ** 6 // args.processes)
    agent = acer.DiscreteACER(model, opt, t_max=args.t_max, gamma=0.99,
                              replay_buffer=replay_buffer,
                              n_times_replay=args.n_times_replay,
                              replay_start_size=args.replay_start_size,
                              beta=args.beta, phi=dqn_phi)

    if args.load:
        agent.load(args.load)

    def make_env(process_idx, test):
        env = ale.ALE(args.rom, use_sdl=args.use_sdl,
                      treat_life_lost_as_terminal=not test)
        if not test:
            misc.env_modifiers.make_reward_clipped(env, -1, 1)

        return env

    if args.demo:
        env = make_env(0, True)
        mean, median, stdev = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev'.format(
            args.eval_n_runs, mean, median, stdev))
    else:
        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_frequency=args.eval_frequency,
            max_episode_len=args.max_episode_len)


if __name__ == '__main__':
    main()
