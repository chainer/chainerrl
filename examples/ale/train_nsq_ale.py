from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import os
import argparse
import random
from logging import getLogger
logger = getLogger(__name__)

import chainer
from chainer import links as L

from chainerrl.action_value import DiscreteActionValue
from chainerrl.links import sequence
from chainerrl.links import dqn_head
from chainerrl.agents import nsq
from chainerrl.envs import ale
from chainerrl.misc import random_seed
from chainerrl.optimizers import rmsprop_async
from chainerrl.experiments.prepare_output_dir import prepare_output_dir
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.misc.init_like_torch import init_like_torch
from chainerrl.experiments.train_agent_async import train_agent_async
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.experiments.evaluator import eval_performance
from chainerrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from chainerrl.explorers.epsilon_greedy import ConstantEpsilonGreedy
from chainerrl import spaces
from dqn_phi import dqn_phi


def main():

    # This prevents numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging
    # logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--use-sdl', action='store_true', default=False)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=4 * 10 ** 6)
    parser.add_argument('--outdir', type=str, default='nsq_output')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--eval-frequency', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    args.outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    def make_env(process_idx, test):
        return ale.ALE(args.rom, use_sdl=args.use_sdl,
                       treat_life_lost_as_terminal=not test)

    sample_env = make_env(0, test=False)
    action_space = sample_env.action_space
    assert isinstance(action_space, spaces.Discrete)

    def make_agent(process_idx):
        q_func = sequence.Sequence(
            dqn_head.NIPSDQNHead(),
            L.Linear(256, action_space.n),
            DiscreteActionValue)
        opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
        opt.setup(q_func)
        # opt.add_hook(chainer.optimizer.GradientClipping(1.0))

        # Random epsilon assignment described in the original paper
        rand = random.random()
        if rand < 0.4:
            epsilon_target = 0.1
        elif rand < 0.7:
            epsilon_target = 0.01
        else:
            epsilon_target = 0.5
        explorer = LinearDecayEpsilonGreedy(
            1, epsilon_target, args.final_exploration_frames,
            action_space.sample)
        # Suppress the explorer logger
        explorer.logger.setLevel(logging.INFO)
        return nsq.NSQ(process_idx, q_func, opt, t_max=5, gamma=0.99,
                       i_target=40000,
                       explorer=explorer, phi=dqn_phi)

    if args.demo:
        env = make_env(0, True)
        agent = make_agent(0)
        mean, median, stdev = eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev'.format(
            args.eval_n_runs, mean, median, stdev))
    else:
        explorer = ConstantEpsilonGreedy(0.05, action_space.sample)
        train_agent_async(
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            make_agent=make_agent,
            profile=args.profile,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_frequency=args.eval_frequency,
            eval_explorer=explorer)

if __name__ == '__main__':
    main()
