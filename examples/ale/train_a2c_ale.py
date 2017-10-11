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
import numpy as np

from chainerrl.agents import a2c
from chainerrl.envs import ale
from chainerrl.envs.vec_env import VectorEnv
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policy
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_function

from dqn_phi import dqn_phi


def phi(screens):
    """Phi (feature extractor) of DQN for ALE

    Args:
      screens: List of N screen objects. Each screen object must be
      numpy.ndarray whose dtype is numpy.uint8.
    Returns:
      numpy.ndarray
    """
    if isinstance(screens, list):
        return dqn_phi(screens)
    assert screens.shape[1] == 4
    assert screens[0].dtype == np.uint8
    raw_values = np.asarray(screens, dtype=np.float32)
    # [0,255] -> [0, 1]
    raw_values /= 255.0
    return raw_values


class A2CFF(chainer.ChainList, a2c.A2CModel):

    def __init__(self, n_actions):
        self.action_space = 1

        self.head = links.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)


class A2CLSTM(chainer.ChainList, a2c.A2CModel, RecurrentChainMixin):

    def __init__(self, n_actions):
        self.action_space = 1
        self.head = links.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        self.lstm = L.LSTM(self.head.n_output_channels,
                           self.head.n_output_channels)
        super().__init__(self.head, self.lstm, self.pi, self.v)

    def pi_and_v(self, state):
        h = self.head(state)
        h = self.lstm(h)
        return self.pi(h), self.v(h)


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
    parser.add_argument('--max-episode-len', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--update-steps', type=int, default=5)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-5)
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer alpha')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--max-grad-norm', type=float, default=40,
                        help='value loss coefficient')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.set_defaults(use_sdl=False)
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    if args.use_lstm:
        model = A2CLSTM(n_actions)
    else:
        model = A2CFF(n_actions)
    optimizer = rmsprop_async.RMSpropAsync(lr=args.lr,
                                           eps=args.rmsprop_epsilon,
                                           alpha=args.alpha)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.max_grad_norm))
    if args.weight_decay > 0:
        optimizer.add_hook(NonbiasWeightDecay(args.weight_decay))
    agent = a2c.A2C(model, optimizer, gamma=args.gamma,
                    gpu=args.gpu,
                    num_processes=args.processes,
                    update_steps=args.update_steps,
                    phi=phi,
                    use_gae=args.use_gae,
                    tau=args.tau)

    if args.load:
        agent.load(args.load)

    def make_env(process_idx, test):
        env = ale.ALE(args.rom, use_sdl=args.use_sdl,
                      treat_life_lost_as_terminal=not test)
        if not test:
            misc.env_modifiers.make_reward_clipped(env, -1, 1)
        return env

    sample_env = VectorEnv([
        make_env(i, args.demo) for i in range(args.processes)])

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev: {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        eval_env = make_env(0, True)
        experiments.train_agent_with_evaluation_sync(
            agent=agent, env=sample_env, steps=args.steps,
            log_interval=args.log_interval,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_env=eval_env)
    sample_env.close()


if __name__ == '__main__':
    main()
