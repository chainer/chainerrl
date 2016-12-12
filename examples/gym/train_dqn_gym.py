from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import argparse
import sys

from chainer import optimizers
from chainer import cuda
import gym
gym.undo_logger_setup()
import numpy as np
from gym import spaces

from chainerrl.agents.dqn import DQN
from chainerrl.agents.double_dqn import DoubleDQN
from chainerrl.misc import random_seed
from chainerrl import replay_buffer
from chainerrl.experiments.prepare_output_dir import prepare_output_dir
from chainerrl.misc.init_like_torch import init_like_torch
from chainerrl import q_function
from chainerrl.misc import env_modifiers
from chainerrl.experiments.train_agent import train_agent_with_evaluation
from chainerrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from chainerrl.misc import reward_filter


def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 5)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--replay-start-size', type=int, default=10 ** 4)
    parser.add_argument('--target-update-frequency',
                        type=int, default=10 ** 4)
    parser.add_argument('--update-frequency', type=int, default=16)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 4)
    parser.add_argument('--n-x-layers', type=int, default=10)
    parser.add_argument('--n-hidden-channels', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--render-train', action='store_true')
    parser.add_argument('--render-eval', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-3)
    parser.set_defaults(window_visible=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    def clip_action_filter(a):
        if isinstance(a, cuda.cupy.ndarray):
            a = cuda.to_cpu(a)
        return np.clip(a, action_space.low, action_space.high)

    def make_env(for_eval):
        env = gym.make(args.env)
        env_modifiers.make_timestep_limited(env, env.spec.timestep_limit)
        if isinstance(env.action_space, spaces.Box):
            env_modifiers.make_action_filtered(env, clip_action_filter)
        if not for_eval:
            env_modifiers.make_reward_filtered(
                env, lambda x: x * args.reward_scale_factor)
            # env_modifiers.make_reward_filtered(env, AverageRewardFilter())
            # env_modifiers.make_reward_filtered(
            #     env, reward_filter.NormalizedRewardFilter(scale=0.01))
        if (args.render_eval and for_eval) or (args.render_train and not for_eval):
            env_modifiers.make_rendered(env)
        return env

    env = make_env(for_eval=False)
    timestep_limit = env.spec.timestep_limit
    obs_size = np.asarray(env.observation_space.shape).prod()
    action_space = env.action_space

    args.outdir = prepare_output_dir(args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    if isinstance(action_space, spaces.Box):
        action_size = np.asarray(action_space.shape).prod()
        q_func = q_function.FCSIContinuousQFunction(
            obs_size, action_size, 100, 2, action_space)
    else:
        n_actions = action_space.n
        q_func = q_function.FCSIQFunction(obs_size, n_actions, 100, 2)
    init_like_torch(q_func)

    # Use the same hyper parameters as the Nature paper's
    # opt = optimizers.RMSpropGraves(
    #     lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)
    opt = optimizers.Adam(eps=1e-2)

    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(5 * 10 ** 5)

    def phi(obs):
        return obs.astype(np.float32)

    def random_action():
        a = action_space.sample()
        if isinstance(a, np.ndarray):
            a = a.astype(np.float32)
        return a

    explorer = LinearDecayEpsilonGreedy(
        args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
        random_action)
    agent = DoubleDQN(q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
                      explorer=explorer, replay_start_size=args.replay_start_size,
                      target_update_frequency=args.target_update_frequency,
                      update_frequency=args.update_frequency,
                      phi=phi, minibatch_size=args.minibatch_size)
    agent.logger.setLevel(logging.DEBUG)

    if len(args.model) > 0:
        agent.load(args.model)

    eval_env = make_env(for_eval=True)

    train_agent_with_evaluation(
        agent=agent, env=env, steps=args.steps,
        eval_n_runs=args.eval_n_runs, eval_frequency=args.eval_frequency,
        outdir=args.outdir, eval_env=eval_env,
        max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
