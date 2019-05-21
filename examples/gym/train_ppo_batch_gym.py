"""An example of training PPO against OpenAI Gym Envs.

This script is an example of training a PPO agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_ppo_gym.py --env CartPole-v0
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import functools

import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.spaces
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents import PPO
from chainerrl import experiments
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 6)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--standardize-advantages', action='store_true')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--window-size', type=int, default=100)

    parser.add_argument('--update-interval', type=int, default=2048)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--entropy-coef', type=float, default=0.0)
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test)
             for idx, env in enumerate(range(args.num_envs))])

    # Only for getting timesteps, and obs-action spaces
    sample_env = gym.make(args.env)
    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5)

    winit_last = chainer.initializers.LeCunNormal(1e-2)

    # Switch policy types accordingly to action space types
    if isinstance(action_space, gym.spaces.Discrete):
        n_actions = action_space.n
        policy = chainer.Sequential(
            L.Linear(None, 64),
            F.tanh,
            L.Linear(None, 64),
            F.tanh,
            L.Linear(None, n_actions, initialW=winit_last),
            chainerrl.distribution.SoftmaxDistribution,
        )
    elif isinstance(action_space, gym.spaces.Box):
        action_size = action_space.low.size
        policy = chainer.Sequential(
            L.Linear(None, 64),
            F.tanh,
            L.Linear(None, 64),
            F.tanh,
            L.Linear(None, action_size, initialW=winit_last),
            chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type='diagonal',
                var_func=lambda x: F.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )
    else:
        print("""\
This example only supports gym.spaces.Box or gym.spaces.Discrete action spaces.""")  # NOQA
        return

    vf = chainer.Sequential(
        L.Linear(None, 64),
        F.tanh,
        L.Linear(None, 64),
        F.tanh,
        L.Linear(None, 1),
    )

    # Combine a policy and a value function into a single model
    model = chainerrl.links.Branched(policy, vf)

    opt = chainer.optimizers.Adam(alpha=args.lr, eps=1e-5)
    opt.setup(model)
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))
    agent = PPO(model, opt,
                obs_normalizer=obs_normalizer,
                gpu=args.gpu,
                update_interval=args.update_interval,
                minibatch_size=args.batchsize, epochs=args.epochs,
                clip_eps_vf=None, entropy_coef=args.entropy_coef,
                standardize_advantages=args.standardize_advantages,
                )

    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_batch_env(True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            agent.optimizer.alpha = value

        lr_decay_hook = experiments.LinearInterpolationHook(
            args.steps, args.lr, 0, lr_setter)

        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            return_window_size=args.window_size,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=False,
            step_hooks=[
                lr_decay_hook,
            ],
        )


if __name__ == '__main__':
    main()
