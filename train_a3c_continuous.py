import argparse
import multiprocessing as mp

import chainer
from chainer import links as L
from chainer import functions as F
import cv2
import gym
import numpy as np

import policy
import v_function
import dqn_head
import a3c
import random_seed
import rmsprop_async
from init_like_torch import init_like_torch
import run_a3c
import env_modifiers


def phi(obs):
    return obs.astype(np.float32)
    # resized = cv2.resize(obs.image_buffer, (84, 84))
    # return resized.transpose(2, 0, 1).astype(np.float32) / 255


class A3CLSTMGaussian(chainer.ChainList, a3c.A3CModel):

    def __init__(self, obs_size, action_size, hidden_size, lstm_size):
        self.pi_head = L.Linear(obs_size, hidden_size)
        self.v_head = L.Linear(obs_size, hidden_size)
        self.pi_lstm = L.LSTM(hidden_size, lstm_size)
        self.v_lstm = L.LSTM(hidden_size, lstm_size)
        self.pi = policy.FCGaussianPolicy(lstm_size, action_size)
        self.v = v_function.FCVFunction(lstm_size)
        super().__init__(self.pi_head, self.v_head,
                         self.pi_lstm, self.v_lstm, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):

        def forward(head, lstm, tail):
            h = F.relu(head(state))
            if keep_same_state:
                prev_h, prev_c = lstm.h, lstm.c
                h = lstm(h)
                lstm.h, lstm.c = prev_h, prev_c
            else:
                h = lstm(h)
            return tail(h)

        pout = forward(self.pi_head, self.pi_lstm, self.pi)
        vout = forward(self.v_head, self.v_lstm, self.v)

        return pout, vout

    def reset_state(self):
        self.pi_lstm.reset_state()
        self.v_lstm.reset_state()

    def unchain_backward(self):
        self.pi_lstm.h.unchain_backward()
        self.pi_lstm.c.unchain_backward()
        self.v_lstm.h.unchain_backward()
        self.v_lstm.c.unchain_backward()


def main():
    import logging
    logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--scenario', type=str, default='basic')
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-4)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--render', action='store_true')
    parser.set_defaults(render=False)
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    def action_filter(a):
        return np.clip(a, sample_env.action_space.low,
                       sample_env.action_space.high)

    def reward_filter(r):
        return r * args.reward_scale_factor

    def make_env(process_idx, test):
        env = gym.make(args.env)
        timestep_limit = env.spec.timestep_limit
        env_modifiers.make_timestep_limited(env, timestep_limit)
        env_modifiers.make_action_filtered(env, action_filter)
        if not test:
            env_modifiers.make_reward_filtered(env, reward_filter)
        if args.render and process_idx == 0 and not test:
            env_modifiers.make_rendered(env)
        return env

    sample_env = gym.make(args.env)
    obs_size = np.asarray(sample_env.observation_space.shape).prod()
    action_size = np.asarray(sample_env.action_space.shape).prod()
    print(obs_size, action_size)

    def model_opt():
        model = A3CLSTMGaussian(obs_size, action_size)
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
