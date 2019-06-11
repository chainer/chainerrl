import os
import logging
import argparse
import statistics
import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers

import gym
import chainerrl
from chainerrl.experiments import collect_demonstrations
from chainerrl.agents.dqfd import DQfD


def main():
    """Parses arguments and runs the example
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gym environment to run the example on')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--final-exploration-frames',
                        type=int, default=4000,
                        help='Timesteps after which we stop ' +
                        'annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.02,
                        help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.01,
                        help='Exploration epsilon used during eval episodes.')
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--steps', type=int, default=15000,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--replay-start-size', type=int, default=1000,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--target-update-interval', type=int, default=200,
                        help='Frequency (in timesteps) at which ' +
                        'the target network is updated.')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Frequency (in timesteps) of evaluation phase.')
    parser.add_argument('--update-interval', type=int, default=1,
                        help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.add_argument('--num-step-return', type=int, default=3)
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    # parser.add_argument('--render', action='store_true', default=False,
    # help='Render env states in a GUI window.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument("--replay-buffer-size", type=int, default=50000,
                        help="Size of replay buffer (Excluding demonstrations)")
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument("--save-demo-trajectories", action="store_true",
                        default=False)

    # Parameters for the FC/Dense network.
    parser.add_argument("--n-hidden-channels", type=int, default=64,
                        help="Number of hidden units in the FC network")
    parser.add_argument("--n-hidden-layers", type=int, default=2,
                        help="Number of hidden layers in the FC network")

    # DQfD specific parameters for loading and pretraining.
    parser.add_argument('--expert-demo-path', type=str, default=None)
    parser.add_argument('--n-pretrain-steps', type=int, default=1500)
    parser.add_argument('--demo-supervised-margin', type=float, default=0.8)
    parser.add_argument('--demo-sample-ratio', type=float, default=0.2)
    parser.add_argument('--loss-coeff-l2', type=float, default=1e-5)
    args = parser.parse_args()

    assert args.expert_demo_path is not None

    import logging
    logging.basicConfig(level=args.logging_level)

    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    chainerrl.misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = chainerrl.experiments.prepare_output_dir(
        args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        """Makes and seeds the environment
        """
        env = gym.make(args.env)

        env_seed = test_seed if test else train_seed
        env.seed(env_seed)
        return env

    env = make_env(train_seed)
    eval_env = make_env(test_seed)

    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        ndim_obs=env.observation_space.low.size,
        n_actions=env.action_space.n,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=args.n_hidden_layers,
        nonlinearity=F.relu)

    if args.noisy_net_sigma is not None:
        links.to_factorized_noisy(q_func)
        # Turn off explorer
        explorer = chainerrl.explorers.Greedy()
    else:
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
            1.0, args.final_epsilon,
            args.final_exploration_frames,
            lambda: np.random.randint(env.action_space.n))

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(env.observation_space.sample()[None])],
        os.path.join(args.outdir, 'model'))

    # Use the Nature paper's hyperparameters
    # opt = optimizers.RMSpropGraves(
    # lr=args.lr, alpha=0.95, momentum=0.0, eps=1e-3)

    opt = chainer.optimizers.Adam(args.lr)
    opt.setup(q_func)

    # Anneal beta from beta0 to 1 throughout training
    betasteps = args.steps / args.update_interval
    rbuf = chainerrl.replay_buffer.PrioritizedReplayBuffer(
        args.replay_buffer_size, alpha=0.6,
        beta0=0.4, betasteps=betasteps,
        num_steps=args.num_step_return)

    # Demo rbuff is set to unlimited capacity.
    # TODO: How do we do this annealing?
    betasteps = args.n_pretrain_steps / args.update_interval
    demo_rbuf = chainerrl.replay_buffer.PrioritizedReplayBuffer(
        capacity=None, alpha=0.6,
        beta0=0.4, betasteps=betasteps,
        num_steps=args.num_step_return)

    n_transitions = 0
    # Fill the demo buffer with expert transitions
    with chainer.datasets.open_pickle_dataset(args.expert_demo_path) as dataset:
        for transition in dataset:
            (obs, a, r, new_obs, done, info) = transition
            n_transitions += 1
            demo_rbuf.append(state=obs,
                             action=a,
                             reward=r,
                             next_state=new_obs,
                             next_action=None,
                             is_state_terminal=done)
            if done:
                demo_rbuf.stop_current_episode()

    print("Demo buffer loaded with", len(demo_rbuf),
          "/", n_transitions, "transitions")

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32)

    agent = DQfD(q_func, opt, rbuf, demo_rbuf, gpu=args.gpu, gamma=0.99,
                 explorer=explorer, n_pretrain_steps=args.n_pretrain_steps,
                 demo_supervised_margin=args.demo_supervised_margin,
                 demo_sample_ratio=args.demo_sample_ratio,
                 replay_start_size=args.replay_start_size,
                 target_update_interval=args.target_update_interval,
                 clip_delta=args.clip_delta,
                 update_interval=args.update_interval,
                 batch_accumulator='sum',
                 phi=phi, minibatch_size=args.minibatch_size)

    if args.load:
        agent.load(args.load)

    if args.demo and args.save_demo_trajectories:
        collect_demonstrations(agent, eval_env, steps=None,
                               episodes=args.eval_n_runs, outdir=args.outdir)
    else:
        chainerrl.experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            eval_env=eval_env,)


if __name__ == "__main__":
    main()
