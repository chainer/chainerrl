from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

import gym
#gym.undo_logger_setup()  # NOQA
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np

import sys
sys.path.insert(0, ".")
from chainerrl import q_functions

import matplotlib as mpl
mpl.use('TkAgg')

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DuelingDQN
from chainerrl import replay_buffer

from chain_env import ChainEnv
from grid_env import GridEnv
from myseq import MySequence

import examples.ale.atari_wrappers as atari_wrappers

def parse_activation(activation_str):
    if activation_str == 'relu':
        return F.relu
    elif activation_str == 'elu':
        return F.elu
    elif activation_str == 'lrelu':
        return F.leaky_relu
    else:
        raise RuntimeError(
            'Not supported activation: {}'.format(activation_str))


def parse_arch(arch, n_actions, activation):
    if arch == 'nature':
        return links.Sequence(
            links.NatureDQNHead(activation=activation),
            L.Linear(512, n_actions),
            DiscreteActionValue)
    elif arch == 'nips':
        return links.Sequence(
            links.NIPSDQNHead(activation=activation),
            L.Linear(256, n_actions),
            DiscreteActionValue)
    elif arch == 'dueling':
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))


def parse_agent(agent):
    return {'DQN': agents.DQN,
            'DoubleDQN': agents.DoubleDQN,
            'SARSA': agents.SARSA,
            'ExpectedSARSA': agents.ExpectedSARSA,
            'PAL': agents.PAL}[agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--use-sdl', action='store_true', default=False)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=1000)
    parser.add_argument('--init-epsilon', type=float, default=1.0)
    parser.add_argument('--final-epsilon', type=float, default=0.1)
    parser.add_argument('--eval-epsilon', type=float, default=0.05)
    parser.add_argument('--arch', type=str, default='nature',
                        choices=['nature', 'nips', 'dueling'])
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--buffer-size', type=int, default=1000)
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--max-episode-len', type=int,
                        default=200,  # 5 minutes with 60/4 fps
                        help='Maximum number of steps for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=100)
    parser.add_argument('--target-update-interval',
                        type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--agent', type=str, default='DQN',
                        choices=['DQN', 'DoubleDQN', 'PAL', 'SARSA', 'ExpectedSARSA'])
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--video', action='store_true', default=False)
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')

    parser.add_argument('--avg', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--table-lr', type=float, default=0.01)
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--noise-constant', type=float, default=-1)
    parser.add_argument('--prop', action='store_true', default=False)
    parser.add_argument('--adam', action='store_true', default=True)
    parser.add_argument('--orig-noise', action='store_true', default=False)
    parser.add_argument('--last-noise', type=int, default=0)
    parser.add_argument('--res', type=int, default=20)
    parser.add_argument('--entropy-coef', type=float, default=0)
    parser.add_argument('--sigma-gamma', type=float, default=0.99)
    parser.add_argument('--noise-coef', type=float, default=1)
    parser.add_argument('--reward-scale-factor', type=float, default=1)
    parser.add_argument('--init-method', type=str, default='/out')
    parser.add_argument('--sigmanet', type=str, default='100,100')
    parser.add_argument('--one-sigma', action='store_true', default=False)
    parser.add_argument('--share-net', action='store_true', default=False)
    parser.add_argument('--swingup', action='store_true', default=False)
    parser.add_argument('--share-gradient', action='store_true', default=False)

    parser.add_argument('--force', type=float, default=0.001)
    parser.add_argument('--scale-sigma', type=float, default=1)
    parser.add_argument('--min-sigma', type=float, default=0)
    parser.add_argument('--reward-noise', type=float, default=0)
    parser.add_argument('--state-noise', type=float, default=0)

    parser.add_argument('--trap', action='store_true', default=False)
    parser.add_argument('--fixed-sigma', action='store_true', default=False)
    parser.add_argument('--table-sigma', action='store_true', default=False)
    parser.add_argument('--head', action='store_true', default=False)
    parser.add_argument('--noisy-y', action='store_true', default=False)
    parser.add_argument('--noisy-t', action='store_true', default=False)
    parser.add_argument('--save-img', action='store_true', default=False)
    parser.add_argument('--no-greedy', action='store_true', default=False)
    parser.add_argument('--table-noise', action='store_true', default=False)

    parser.add_argument('--use-table', action='store_true', default=False)
    parser.add_argument('--no-nn', action='store_true', default=False)

    parser.add_argument('--env', type=str, default='grid')

    parser.add_argument('--len', type=int, default=20)
    args = parser.parse_args()

    head = args.head

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    chain_len = args.len

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        if args.env == "chain":
            env = ChainEnv(chain_len)
        elif args.env == "grid":
            env = GridEnv(args.outdir, chain_len, save_img=args.save_img,)
        elif args.env == "pole":
            env = gym.make("CartPole-v0")
            e = env.unwrapped
            import math


            if args.swingup:
                e.action_space = spaces.Discrete(3)

                def step(action):
                    state = e.state
                    x, x_dot, theta, theta_dot = state
                    if action==0:
                        force = e.force_mag
                        reward = -0.1
                    elif action==1:
                        force = -e.force_mag
                        reward = -0.1
                    else:
                        force = 0
                        reward = 0
                    costheta = math.cos(theta)
                    sintheta = math.sin(theta)
                    temp = (force + e.polemass_length * theta_dot * theta_dot * sintheta) / e.total_mass
                    thetaacc = (e.gravity * sintheta - costheta* temp) / (e.length * (4.0/3.0 - e.masspole * costheta * costheta / e.total_mass))
                    xacc  = temp - e.polemass_length * thetaacc * costheta / e.total_mass
                    if e.kinematics_integrator == 'euler':
                        x  = x + e.tau * x_dot
                        x_dot = x_dot + e.tau * xacc
                        theta = theta + e.tau * theta_dot
                        theta_dot = theta_dot + e.tau * thetaacc
                    else: # semi-implicit euler
                        x_dot = x_dot + e.tau * xacc
                        x  = x + e.tau * x_dot
                        theta_dot = theta_dot + e.tau * thetaacc
                        theta = theta + e.tau * theta_dot
                    e.state = (x,x_dot,theta,theta_dot)

                    # until max steps
                    done = False

                    if costheta > 0.95 and abs(x) < 0.1 and abs(theta_dot) < 1 and abs(x_dot) < 1:
                        reward = 1

                    return np.array(e.state), reward, done, {}

                def reset():
                    #self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
                    e.state = e.np_random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.02, 0.02, 0.02, 0.02]))
                    e.steps_beyond_done = None
                    return np.array(e.state)

                e.step = step
                e.reset = reset

            #e.reset = reset
            env._max_episode_steps = args.max_episode_len
        elif args.env == "acrobot":
            env = gym.make("Acrobot-v1")
        elif args.env == "car":
            env = gym.make("MountainCar-v0")
            import math

            e = env.unwrapped

            def step(action):
                assert e.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

                position, velocity = e.state
                velocity += (action-1)*args.force + math.cos(3*position)*(-0.0025)
                velocity = np.clip(velocity, -e.max_speed, e.max_speed)
                position += velocity
                position = np.clip(position, e.min_position, e.max_position)
                if (position==e.min_position and velocity<0): velocity = 0

                done = bool(position >= e.goal_position)
                goal = done

                if not test:
                    reward = -1.0 + np.random.normal() * args.reward_noise
                else:
                    reward = -1.0

                if not test:
                    e.state = (position, velocity)
                    a = position + np.random.normal() * args.state_noise * (e.max_position - e.min_position)
                    b = velocity + np.random.normal() * args.state_noise * e.max_speed * 2.0
                    state = np.array((a, b))
                else:
                    e.state = (position, velocity)
                    state = np.array(e.state)

                return state, reward, done, {'goal': goal}

            e.step = step

            env._max_episode_steps = args.max_episode_len

        elif "NoFrame" in args.env:
            env = atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(args.env),
                episode_life=not test,
                clip_rewards=not test,
                fire_reset=True)
            env.seed(int(env_seed))
        else:
            env = gym.make(args.env)

        if not test:
            misc.env_modifiers.make_reward_filtered(
                env, lambda x: x * args.reward_scale_factor)

        if args.render:
            misc.env_modifiers.make_rendered(env, mode='rgb_array')

        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)

        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n

    activation = parse_activation(args.activation)
    #q_func = q_functions.FCStateQFunctionWithDiscreteAction(
    #    n_obs, n_actions,
    #    n_hidden_channels=32,
    #    n_hidden_layers=2)#

    if args.no_nn:
        q_func = None
    else:
        if "NoFrame" in args.env:
            q_func = MySequence(None, n_actions, head, args.avg, scale_sigma=args.scale_sigma)
        else:
            try:
                n_obs = env.observation_space.n
            except:
                n_obs = env.observation_space.shape[0]
            q_func = MySequence(n_obs, n_actions, head, args.avg, args.sigmanet,
                shared=args.share_net, scale_sigma=args.scale_sigma, share_grad=args.share_gradient)

    """
    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((n_obs), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'diagram'))
    """

    if args.adam:
        opt = optimizers.Adam(args.lr)
    else:
        # Use the same hyper parameters as the Nature paper's
        opt = optimizers.RMSpropGraves(
            lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)

    rbuf = replay_buffer.ReplayBuffer(args.buffer_size)

    explorer = explorers.LinearDecayEpsilonGreedy(
        args.init_epsilon, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    entropy = None
    if args.noisy_net_sigma is not None and args.noisy_net_sigma > 0:
        entropy = links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma, constant=args.noise_constant,
            prev=args.orig_noise, noise_coef=args.noise_coef, init_method=args.init_method)
        # Turn off explorer
        if not args.no_greedy:
            explorer = explorers.Greedy()

        if args.last_noise > 0:
            for e in entropy[:-args.last_noise]:
                e.off = True
    elif args.head:
        if not args.no_greedy:
            explorer = explorers.Greedy()

    """
    print(n_obs)
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((n_obs), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'diagram2'))
    """

    if not args.no_nn:
        opt.setup(q_func)

    def phi(x):
        # Feature extractor
        if "NoFrame" in args.env:
            return np.asarray(x, dtype=np.float32) / 255.0
        else:
            return np.asarray(x, dtype=np.float32)

    Agent = parse_agent(args.agent)

    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=args.target_update_interval,
                  clip_delta=args.clip_delta,
                  update_interval=args.update_interval,
                  batch_accumulator='sum',
                  minibatch_size=args.minibatch_size,
                  phi=phi, entropy=entropy, entropy_coef=args.entropy_coef,
                  vis=env, noisy_y=args.noisy_y, noisy_t=args.noisy_t,
                  plot=args.save_img,
                  head=head, use_table=args.use_table,
                  table_lr=args.table_lr,
                  samples=args.samples,
                  env=args.env,
                  video=args.video,
                  table_sigma=args.table_sigma,
                  scale_sigma=args.scale_sigma,
                  min_sigma=args.min_sigma,
                  no_nn=args.no_nn,
                  outdir=args.outdir,
                  algo=args.agent,
                  sigma_gamma=args.sigma_gamma,
                  one_sigma=args.one_sigma,
                  fixed_sigma=args.fixed_sigma,
                  table_noise=args.table_noise,
                  gym_env=env,
                  res=args.res)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        # In testing DQN, randomly select 5% of actions
        eval_explorer = explorers.Greedy()
        #explorers.ConstantEpsilonGreedy(
        #    args.eval_epsilon, lambda: np.random.randint(n_actions))

        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_explorer=eval_explorer,
            max_episode_len=args.max_episode_len,
            eval_env=eval_env,
            save_best_so_far_agent=False,
        )


if __name__ == '__main__':
    main()
