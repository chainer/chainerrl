import argparse
import sys

from chainer import optimizers
from chainer import cuda
import gym
import numpy as np

sys.path.append('../drill')
from dqn import DQN
import random_seed
import replay_buffer
from prepare_output_dir import prepare_output_dir
from init_like_torch import init_like_torch
import q_function
import env_modifiers
import run_dqn


def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 6)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--replay-start-size', type=int, default=10 ** 4)
    parser.add_argument('--target-update-frequency',
                        type=int, default=10 ** 4)
    parser.add_argument('--update-frequency', type=int, default=16)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 5)
    parser.add_argument('--n-x-layers', type=int, default=10)
    parser.add_argument('--n-hidden-channels', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-3)
    parser.set_defaults(window_visible=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    def clip_action_filter(a):
        if isinstance(a, cuda.cupy.ndarray):
            a = cuda.to_cpu(a)
        return np.clip(a, action_space.low, action_space.high)

    def reward_filter(r):
        return r * args.reward_scale_factor

    def make_env():
        env = gym.make(args.env)
        timestep_limit = env.spec.timestep_limit
        env_modifiers.make_timestep_limited(env, timestep_limit)
        env_modifiers.make_action_filtered(env, clip_action_filter)
        env_modifiers.make_reward_filtered(env, reward_filter)
        if args.render:
            env_modifiers.make_rendered(env)

        def __exit__(self, *args):
            pass
        env.__exit__ = __exit__
        return env

    sample_env = gym.make(args.env)
    # timestep_limit = sample_env.spec.timestep_limit
    obs_size = np.asarray(sample_env.observation_space.shape).prod()
    obs_space = sample_env.observation_space
    action_size = np.asarray(sample_env.action_space.shape).prod()
    action_space = sample_env.action_space
    print(obs_size, action_size)
    del sample_env

    args.outdir = prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    q_func = q_function.FCSIContinuousQFunction(
        obs_size, action_size, 100, 2, action_space)
    init_like_torch(q_func)

    # Use the same hyper parameters as the Nature paper's
    # opt = optimizers.RMSpropGraves(
    #     lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)
    opt = optimizers.Adam()

    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(5 * 10 ** 5)

    def phi(obs):
        return obs.astype(np.float32)

    agent = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
                epsilon=0.1, replay_start_size=args.replay_start_size,
                target_update_frequency=args.target_update_frequency,
                update_frequency=args.update_frequency,
                phi=phi, minibatch_size=args.minibatch_size)

    if len(args.model) > 0:
        agent.load_model(args.model)

    run_dqn.run_dqn(
        agent=agent, make_env=make_env, phi=phi, steps=args.steps,
        eval_n_runs=args.eval_n_runs, eval_frequency=args.eval_frequency,
        gpu=args.gpu, outdir=args.outdir,
        final_exploration_steps=args.final_exploration_steps)


if __name__ == '__main__':
    main()
