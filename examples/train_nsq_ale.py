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
import sys
from logging import getLogger
logger = getLogger(__name__)

import chainer

sys.path.append('..')
from links import fc_tail_q_function
from links import dqn_head
from agents import nstep_q_learning
from envs import ale
import random_seed
import async
import rmsprop_ones
from prepare_output_dir import prepare_output_dir


def main():

    # This line makes execution much faster, I don't know why
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--steps', type=int, default=10 ** 6)
    parser.add_argument('--use-sdl', action='store_true')
    parser.add_argument('--final-exploration-frames', type=int, default=1e6)
    parser.add_argument('--outdir', type=str, default=None)
    parser.set_defaults(use_sdl=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    def agent_func(process_idx):
        head = dqn_head.NIPSDQNHead()
        q_func = fc_tail_q_function.FCTailQFunction(
            head, 256, n_actions=n_actions)
        opt = rmsprop_ones.RMSpropOnes(lr=1e-3, eps=1e-4)
        opt.setup(q_func)
        opt.add_hook(chainer.optimizer.GradientClipping(1.0))
        return nstep_q_learning.NStepQLearning(q_func, opt, 5, 0.99, 1.0,
                                               i_target=40000 // args.processes)

    def env_func(process_idx):
        return ale.ALE(args.rom, use_sdl=args.use_sdl)

    def run_func(process_idx, agent, env):

        # Random epsilon assignment described in the original paper
        rand = random.random()
        if rand < 0.4:
            epsilon_target = 0.1
        elif rand < 0.7:
            epsilon_target = 0.01
        else:
            epsilon_target = 0.5

        total_r = 0
        episode_r = 0

        for i in range(args.steps):

            total_r += env.reward
            episode_r += env.reward

            if agent.epsilon > epsilon_target:
                agent.epsilon -= (1 - epsilon_target) / \
                    args.final_exploration_frames

            action = agent.act(env.state, env.reward, env.is_terminal)

            if env.is_terminal:
                if process_idx == 0:
                    logger.debug('{} i:{} epsilon:{} episode_r:{}'.format(
                        outdir, i, agent.epsilon, episode_r))
                episode_r = 0
                env.initialize()
            else:
                env.receive_action(action)

        if process_idx == 0:
            print(logger.debug('{} pid:{}, total_r:{}'.format(outdir, os.getpid(), total_r)))

    async.run_async(args.processes, agent_func, env_func, run_func)


if __name__ == '__main__':
    main()
