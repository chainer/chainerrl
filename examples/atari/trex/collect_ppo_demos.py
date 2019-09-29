"""An example of training PPO against OpenAI Gym Atari Envs.

This script collects demos of a PPO agent on Atari envs.

To collect demos for  PPO for 10M timesteps on Breakout, run:
    python collect_demos_ppo.py
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

import chainer
from chainer import functions as F
from chainer import links as L
import numpy as np

import chainerrl
from chainerrl.agents import PPO
from chainerrl import experiments
from chainerrl import misc
from chainerrl.wrappers import atari_wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',
                        help='Gym Env ID.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID. Set to -1 to use CPUs only.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Run demo episodes, not training.')
    parser.add_argument('--load', type=str, required=True,
                        help='Directory path where agents are stored'
                             ' if it is a non-empty string.')
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--checkpoint-frequency', type=int,
                        default=None,
                        help='Frequency with which networks were checkpointed.')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env():
        test = True
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test,
            flicker=False,
            frame_stack=True,
        )
        env.seed(env_seed)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env()

    print('Observation space', env.observation_space)
    print('Action space', env.action_space)
    n_actions = env.action_space.n

    winit_last = chainer.initializers.LeCunNormal(1e-2)
    model = chainer.Sequential(
        L.Convolution2D(None, 32, 8, stride=4),
        F.relu,
        L.Convolution2D(None, 64, 4, stride=2),
        F.relu,
        L.Convolution2D(None, 64, 3, stride=1),
        F.relu,
        L.Linear(None, 512),
        F.relu,
        chainerrl.links.Branched(
            chainer.Sequential(
                L.Linear(None, n_actions, initialW=winit_last),
                chainerrl.distribution.SoftmaxDistribution,
            ),
            L.Linear(None, 1),
        )
    )

    # Draw the computational graph and save it in the output directory.
    fake_obss = np.zeros(
        env.observation_space.shape, dtype=np.float32)[None]
    fake_out = model(fake_obss)
    chainerrl.misc.draw_computational_graph(
        [fake_out], os.path.join(args.outdir, 'model'))

    opt = chainer.optimizers.Adam()
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(0.5))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255


    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        phi=phi,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-2,
    )

    agent.load(args.load)

    # saves demos to outdir/demos.pickle
    experiments.collect_demonstrations(agent=agent,
                                       env=env,
                                       steps=None,
                                       episodes=1,
                                       outdir=args.outdir,
                                       max_episode_len=None)


if __name__ == '__main__':
    main()
