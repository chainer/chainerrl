import argparse
import functools
import os

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DuelingDQN
from chainerrl import replay_buffer

from chainerrl.wrappers import atari_wrappers


class SingleSharedBias(chainer.Chain):
    """Single shared bias used in the Double DQN paper.

    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.

    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.bias = chainer.Parameter(0, shape=1)

    def __call__(self, x):
        return x + F.broadcast_to(self.bias, x.shape)


def parse_arch(arch, n_actions):
    if arch == 'nature':
        return links.Sequence(
            links.NatureDQNHead(),
            L.Linear(512, n_actions),
            DiscreteActionValue)
    elif arch == 'doubledqn':
        return links.Sequence(
            links.NatureDQNHead(),
            L.Linear(512, n_actions, nobias=True),
            SingleSharedBias(),
            DiscreteActionValue)
    elif arch == 'nips':
        return links.Sequence(
            links.NIPSDQNHead(),
            L.Linear(256, n_actions),
            DiscreteActionValue)
    elif arch == 'dueling':
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))


def parse_agent(agent):
    return {'DQN': agents.DQN,
            'DoubleDQN': agents.DoubleDQN,
            'PAL': agents.PAL}[agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--final-epsilon', type=float, default=0.01)
    parser.add_argument('--eval-epsilon', type=float, default=0.001)
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--arch', type=str, default='doubledqn',
                        choices=['nature', 'nips', 'dueling', 'doubledqn'])
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7)
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',
                        type=int, default=3 * 10 ** 4)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--agent', type=str, default='DoubleDQN',
                        choices=['DQN', 'DoubleDQN', 'PAL'])
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate')
    parser.add_argument('--prioritized', action='store_true', default=False,
                        help='Use prioritized experience replay.')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--n-step-return', type=int, default=1)
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
            frame_stack=False,
        )
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        env.seed(env_seed)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        vec_env = chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test)
             for idx, env in enumerate(range(args.num_envs))])
        vec_env = chainerrl.wrappers.VectorFrameStack(vec_env, 4)
        return vec_env

    sample_env = make_env(0, test=False)

    n_actions = sample_env.action_space.n
    q_func = parse_arch(args.arch, n_actions)

    if args.noisy_net_sigma is not None:
        links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as the Nature paper's
    opt = optimizers.RMSpropGraves(
        lr=args.lr, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            10 ** 6, alpha=0.6, beta0=0.4, betasteps=betasteps,
            num_steps=args.n_step_return,
        )
    else:
        rbuf = replay_buffer.ReplayBuffer(
            10 ** 6, num_steps=args.n_step_return)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = parse_agent(args.agent)
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=args.target_update_interval,
                  clip_delta=args.clip_delta,
                  update_interval=args.update_interval,
                  batch_accumulator='sum',
                  phi=phi)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=make_batch_env(test=True),
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            log_interval=1000,
        )


if __name__ == '__main__':
    main()
