import argparse
import os

from chainer import links as L
from chainer import optimizers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl import replay_buffer

from chainerrl.wrappers import atari_wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',
                        help='OpenAI Atari domain to perform algorithm on.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--load', type=str, default=None, required=True)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7,
                        help='Total number of demo timesteps to collect')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env():
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=None),
            episode_life=False,
            clip_rewards=False)
        env.seed(int(args.seed))
        # Randomize actions like epsilon-greedy
        env = chainerrl.wrappers.RandomizeAction(env, 0.01)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env()

    n_actions = env.action_space.n
    q_func = links.Sequence(
        links.NatureDQNHead(),
        L.Linear(512, n_actions),
        DiscreteActionValue)

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # The optimizer and replay buffer are dummy variables required by agent
    opt = optimizers.RMSpropGraves()
    opt.setup(q_func)
    rbuf = replay_buffer.ReplayBuffer(1)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = agents.DQN
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=None, replay_start_size=1,
                  minibatch_size=1,
                  target_update_interval=None,
                  clip_delta=True,
                  update_interval=4,
                  phi=phi)

    agent.load(args.load)

    # saves demos to outdir/demos.pickle
    experiments.collect_demonstrations(agent=agent,
                                       env=env,
                                       steps=args.steps,
                                       episodes=None,
                                       outdir=args.outdir,
                                       max_episode_len=None)


if __name__ == '__main__':
    main()
