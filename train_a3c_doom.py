import argparse
import multiprocessing as mp

import chainer
from chainer import links as L
from chainer import functions as F
import cv2
import numpy as np

import policy
import v_function
import dqn_head
import a3c
import random_seed
import rmsprop_async
from init_like_torch import init_like_torch
import run_a3c
import doom_env


def phi(obs):
    resized = cv2.resize(obs.image_buffer, (84, 84))
    return resized.transpose(2, 0, 1).astype(np.float32) / 255


class A3CFF(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead(n_input_channels=3)
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
        self.head = dqn_head.NIPSDQNHead(n_input_channels=3)
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
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--scenario', type=str, default='basic')
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--window-visible', action='store_true')
    parser.set_defaults(window_visible=False)
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    # Simultaneously launching multiple vizdoom processes makes program stuck,
    # so use the global lock
    env_lock = mp.Lock()

    def make_env(process_idx, test):
        with env_lock:
            return doom_env.DoomEnv(window_visible=args.window_visible,
                                    scenario=args.scenario)

    n_actions = 3

    def model_opt():
        if args.use_lstm:
            model = A3CLSTM(n_actions)
        else:
            model = A3CFF(n_actions)
        opt = rmsprop_async.RMSpropAsync(lr=args.lr, eps=1e-1, alpha=0.99)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        return model, opt

    run_a3c.run_a3c(args.processes, make_env, model_opt, phi, t_max=args.t_max,
                    beta=args.beta, profile=args.profile, steps=args.steps,
                    eval_frequency=args.eval_frequency,
                    eval_n_runs=args.eval_n_runs, args=args)


if __name__ == '__main__':
    main()
