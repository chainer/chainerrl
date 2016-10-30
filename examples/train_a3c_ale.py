from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super
from builtins import str
from future import standard_library
standard_library.install_aliases()
import argparse
import os
import sys

import chainer
from chainer import links as L

sys.path.append('..')
from chainerrl import policy
from chainerrl import v_function
from chainerrl.links import dqn_head
from chainerrl.agents import a3c
from chainerrl.envs import ale
from chainerrl.misc import random_seed
from chainerrl.optimizers import rmsprop_async
from chainerrl.experiments.prepare_output_dir import prepare_output_dir
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.misc.init_like_torch import init_like_torch
from chainerrl.experiments.train_agent_async import train_agent_async
from dqn_phi import dqn_phi


class A3CFF(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        return self.pi(out), self.v(out)


class A3CLSTM(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        self.lstm = L.LSTM(self.head.n_output_channels,
                           self.head.n_output_channels)
        super().__init__(self.head, self.lstm, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            out = self.lstm(out)
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            out = self.lstm(out)
        return self.pi(out), self.v(out)

    def reset_state(self):
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()


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
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--use-lstm', action='store_true')
    parser.set_defaults(use_sdl=False)
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    args.outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    def make_agent(process_idx):
        if args.use_lstm:
            model = A3CLSTM(n_actions)
        else:
            model = A3CFF(n_actions)
        opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        if args.weight_decay > 0:
            opt.add_hook(NonbiasWeightDecay(args.weight_decay))
        return a3c.A3C(model, opt, t_max=args.t_max, gamma=0.99,
                       beta=args.beta, process_idx=process_idx, phi=dqn_phi)

    def make_env(process_idx, test):
        return ale.ALE(args.rom, use_sdl=args.use_sdl,
                       treat_life_lost_as_terminal=not test)

    train_agent_async(
        outdir=args.outdir,
        processes=args.processes,
        make_env=make_env,
        make_agent=make_agent,
        profile=args.profile,
        steps=args.steps,
        eval_n_runs=args.eval_n_runs,
        eval_frequency=args.eval_frequency)


if __name__ == '__main__':
    main()
