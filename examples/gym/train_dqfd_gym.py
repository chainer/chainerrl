"""An example of training DQfD for OpenAI gym Environments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
from builtins import *  # NOQA

import chainer
import chainer.functions as F

import chainerrl

from chainerrl import experiments
from chainerrl.agents.dqfd import DQfD, PrioritizedDemoReplayBuffer

from future import standard_library
standard_library.install_aliases()  # NOQA

import gym

import numpy as np


def main():
    """Parses arguments and runs the example
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gym environment to run the example on')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0)
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
    parser.add_argument('--render-train', action='store_true')
    parser.add_argument('--render-eval', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument("--replay-buffer-size", type=int, default=50000,
                        help="Size of replay buffer (w/o demonstrations)")
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument("--save-demo-trajectories", action="store_true",
                        default=False)
    parser.add_argument('--reward-scale-factor', type=float, default=1.0)
    parser.add_argument("--n-hidden-channels", type=int, default=64)
    parser.add_argument("--n-hidden-layers", type=int, default=2)

    # DQfD specific parameters for loading and pretraining.
    parser.add_argument('--expert-demo-path', type=str, default=None)
    parser.add_argument('--n-pretrain-steps', type=int, default=1500)
    parser.add_argument('--demo-supervised-margin', type=float, default=0.8)
    parser.add_argument('--loss-coeff-l2', type=float, default=1e-5)
    parser.add_argument('--loss-coeff-nstep', type=float, default=1.0)
    parser.add_argument('--loss-coeff-supervised', type=float, default=1.0)
    parser.add_argument('--bonus-priority-agent', type=float, default=0.001)
    parser.add_argument('--bonus-priority-demo', type=float, default=1.0)
    parser.add_argument('--priority-error-max', type=float, default=2.0)
    args = parser.parse_args()

    assert args.expert_demo_path is not None, "DQfD needs collected \
                        expert demonstrations"

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

        # Cast observations to float32 because our model uses float32
        # env = chainerrl.wrappers.CastObservationToFloat32(env)
        # if args.monitor:
        # env = gym.wrappers.Monitor(env, args.outdir)
        # if not test:
        # Scale rewards (and thus returns) to a reasonable range so that
        # training is easier
        # env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        # if ((args.render_eval and test) or
        # (args.render_train and not test)):
        # env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        ndim_obs=env.observation_space.low.size,
        n_actions=env.action_space.n,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=args.n_hidden_layers,
        nonlinearity=F.relu)

    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(env.action_space.n))

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(env.observation_space.sample()[None])],
        os.path.join(args.outdir, 'model'))

    opt = chainer.optimizers.Adam(args.lr)
    opt.setup(q_func)
    betasteps = args.steps / args.update_interval
    replay_buffer = PrioritizedDemoReplayBuffer(
        args.replay_buffer_size, alpha=0.6,
        beta0=0.4, betasteps=betasteps, error_max=args.priority_error_max,
        num_steps=args.num_step_return)

    # Fill the demo buffer with expert transitions
    n_demo_transitions = 0
    with chainer.datasets.open_pickle_dataset(args.expert_demo_path) as dset:
        for transition in dset:
            (obs, a, r, new_obs, done, info) = transition
            n_demo_transitions += 1
            replay_buffer.append(state=obs,
                                 action=a,
                                 reward=r,
                                 next_state=new_obs,
                                 next_action=None,
                                 is_state_terminal=done,
                                 demo=True)
            if ("needs_reset" in info and info["needs_reset"]):
                replay_buffer.stop_current_episode(demo=True)
    print("Demo buffer loaded with %d (1 and n-step) transitions from "
          "%d expert demonstration transitions" % (len(replay_buffer),
                                                   n_demo_transitions))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32)

    agent = DQfD(q_func, opt, replay_buffer,
                 gamma=0.99,
                 explorer=explorer,
                 n_pretrain_steps=args.n_pretrain_steps,
                 demo_supervised_margin=args.demo_supervised_margin,
                 bonus_priority_agent=args.bonus_priority_agent,
                 bonus_priority_demo=args.bonus_priority_demo,
                 loss_coeff_nstep=args.loss_coeff_nstep,
                 loss_coeff_supervised=args.loss_coeff_supervised,
                 loss_coeff_l2=args.loss_coeff_l2,
                 gpu=args.gpu,
                 replay_start_size=args.replay_start_size,
                 target_update_interval=args.target_update_interval,
                 clip_delta=args.clip_delta,
                 update_interval=args.update_interval,
                 batch_accumulator='sum',
                 phi=phi, minibatch_size=args.minibatch_size)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        logger = logging.getLogger(__name__)
        evaluator = experiments.Evaluator(agent=agent,
                                          n_steps=None,
                                          n_episodes=args.eval_n_runs,
                                          eval_interval=args.eval_interval,
                                          outdir=args.outdir,
                                          max_episode_len=None,
                                          env=eval_env,
                                          step_offset=0,
                                          save_best_so_far_agent=True,
                                          logger=logger)

        # Evaluate the agent BEFORE training begins
        evaluator.evaluate_and_update_max_score(t=0, episodes=0)
        experiments.train_agent(agent=agent,
                                env=env,
                                steps=args.steps,
                                outdir=args.outdir,
                                max_episode_len=None,
                                step_offset=0,
                                evaluator=evaluator,
                                successful_score=None,
                                step_hooks=[])


if __name__ == "__main__":
    main()
