from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import argparse
import sys

import chainer
from chainer import optimizers
from chainer import cuda
import gym
import numpy as np
from gym import spaces

from chainerrl import policy
from chainerrl import q_function
from chainerrl import replay_buffer
from chainerrl.agents.ddpg import DDPG, DDPGModel
from chainerrl.experiments.evaluator import eval_performance
from chainerrl.experiments.prepare_output_dir import prepare_output_dir
from chainerrl.experiments.train_agent import train_agent_with_evaluation
from chainerrl.explorers.additive_gaussian import AdditiveGaussian
from chainerrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from chainerrl.misc import env_modifiers
from chainerrl.misc import random_seed
from chainerrl.misc.init_like_torch import init_like_torch


def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='out')
    parser.add_argument('--env', type=str, default='Humanoid-v1')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 6)
    parser.add_argument('--start-epsilon', type=float, default=0.4)
    parser.add_argument('--end-epsilon', type=float, default=0.3)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--replay-start-size', type=int, default=5000)
    parser.add_argument('--target-update-frequency',
                        type=int, default=5000)
    parser.add_argument('--update-frequency', type=int, default=1000)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 5)
    #parser.add_argument('--n-x-layers', type=int, default=10)
    #parser.add_argument('--n-hidden-channels', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--minibatch-size', type=int, default=200)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.set_defaults(window_visible=True)
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
        #env.monitor.start('/tmp', force=True, seed=0)
        timestep_limit = env.spec.timestep_limit
        env_modifiers.make_timestep_limited(env, timestep_limit)
        if isinstance(env.action_space, spaces.Box):
            env_modifiers.make_action_filtered(env, clip_action_filter)
        env_modifiers.make_reward_filtered(env, reward_filter)
        if args.render:
            env_modifiers.make_rendered(env)

        def __exit__(self, *args):
            pass
        env.__exit__ = __exit__
        return env

    env = make_env()
    # timestep_limit = sample_env.spec.timestep_limit
    obs_size = np.asarray(env.observation_space.shape).prod()
    action_space = env.action_space

    args.outdir = prepare_output_dir(args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    action_size = np.asarray(action_space.shape).prod()
    model = DDPGModel(
        q_func=q_function.FCBNSAQFunction(
            obs_size, action_size, n_hidden_channels=300, n_hidden_layers=3),
        policy=policy.FCBNDeterministicPolicy(
            obs_size, action_size=action_size,
            n_hidden_layers=2, n_hidden_channels=300,
            min_action=action_space.low, max_action=action_space.high,
            bound_action=True)
    )
    init_like_torch(model['q_function'])
    init_like_torch(model['policy'])
    opt_a = optimizers.Adam(alpha=1e-5)
    opt_c = optimizers.Adam(alpha=1e-3)
    opt_a.setup(model['policy'])
    opt_c.setup(model['q_function'])
    opt_a.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
    opt_c.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')
    # opt_c.add_hook(chainer.optimizer.WeightDecay(1e-4))

    rbuf = replay_buffer.ReplayBuffer(5 * 10 ** 5)

    def phi(obs):
        return obs.astype(np.float32)

    def random_action():
        a = action_space.sample()
        if isinstance(a, np.ndarray):
            a = a.astype(np.float32)
        return a

    explorer = AdditiveGaussian(scale=0.4)
    agent = DDPG(model, opt_a, opt_c, rbuf, gamma=args.gamma,
                 explorer=explorer, replay_start_size=args.replay_start_size,
                 target_update_frequency=args.target_update_frequency,
                 update_frequency=args.update_frequency,
                 n_times_update=10,
                 phi=phi, gpu=args.gpu, minibatch_size=args.minibatch_size)
    agent.logger.setLevel(logging.DEBUG)

    if len(args.load) > 0:
        agent.load(args.load)

    if args.demo:
        mean, median, stdev = eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev'.format(
            args.eval_n_runs, mean, median, stdev))
    else:
        train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_frequency=args.eval_frequency,
            outdir=args.outdir)

if __name__ == '__main__':
    main()
