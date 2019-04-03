from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse

import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents import PPO
from chainerrl import experiments
from chainerrl import misc
from chainerrl.wrappers import atari_wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--eval-epsilon', type=float, default=0.001)
    parser.add_argument('--standardize-advantages', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--update-interval', type=int, default=128 * 8)
    parser.add_argument('--batchsize', type=int, default=32 * 8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--decay-clip-eps', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=10000)
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--adam-eps', type=float, default=1e-8)
    parser.add_argument('--flicker', action='store_true', default=False)
    parser.add_argument('--no-frame-stack', action='store_true', default=False)
    parser.add_argument('--max-grad-norm', type=float, default=.5)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test,
            flicker=args.flicker,
            frame_stack=not args.no_frame_stack,
        )
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        env.seed(env_seed)
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [(lambda: make_env(idx, test))
             for idx, env in enumerate(range(args.num_envs))])

    sample_env = make_env(0, test=False)
    print('Observation space', sample_env.observation_space)
    print('Action space', sample_env.action_space)
    n_actions = sample_env.action_space.n

    winit_last = chainer.initializers.LeCunNormal(1e-2)
    if args.recurrent:
        model = chainerrl.links.StatelessRecurrentSequential(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            F.relu,
            L.Linear(None, 512),
            F.relu,
            L.NStepLSTM(1, 512, 512, 0),
            chainerrl.links.ParallelLink(
                chainer.Sequential(
                    L.Linear(None, n_actions, initialW=winit_last),
                    chainerrl.distribution.SoftmaxDistribution,
                ),
                L.Linear(None, 1, initialW=winit_last),
            )
        )
    else:
        model = chainer.Sequential(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            F.relu,
            L.Linear(None, 512),
            F.relu,
            chainerrl.links.ParallelLink(
                chainer.Sequential(
                    L.Linear(None, n_actions, initialW=winit_last),
                    chainerrl.distribution.SoftmaxDistribution,
                ),
                L.Linear(None, 1, initialW=winit_last),
            )
        )
    opt = chainer.optimizers.Adam(alpha=args.lr, eps=args.adam_eps)
    opt.setup(model)
    if args.max_grad_norm > 0:
        opt.add_hook(chainer.optimizer.GradientClipping(args.max_grad_norm))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = PPO(
        model, opt,
        gpu=args.gpu,
        phi=phi,
        update_interval=args.update_interval,
        minibatch_size=args.batchsize, epochs=args.epochs,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=args.standardize_advantages,
        entropy_coef=1e-2,
        recurrent=args.recurrent,
    )
    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev: {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            agent.optimizer.alpha = value

        step_hooks.append(
            experiments.LinearInterpolationHook(
                args.steps, args.lr, 0, lr_setter))

        if args.decay_clip_eps:
            # Linearly decay the clipping parameter to zero
            def clip_eps_setter(env, agent, value):
                agent.clip_eps = max(value, 1e-8)

            step_hooks.append(
                experiments.LinearInterpolationHook(
                    args.steps, 0.1, 0, clip_eps_setter))

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
            save_best_so_far_agent=False,
            step_hooks=step_hooks,
        )


if __name__ == '__main__':
    main()
