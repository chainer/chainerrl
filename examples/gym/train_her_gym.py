from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import sys

import chainer
from chainer import optimizers
import gym
from gym import spaces
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents.ddpg import DDPG
from chainerrl.agents.ddpg import DDPGModel
from chainerrl import experiments
from chainerrl import explorer
from chainerrl import explorers
from chainerrl import misc
from chainerrl import policy
from chainerrl import q_functions
from chainerrl import replay_buffer
import os 


class HEREnvWrapper(gym.Wrapper):

    def __init__(self, env, outdir):
        super(HEREnvWrapper, self).__init__(env)
        self.outdir = outdir

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            with open(os.path.join(self.outdir, 'successes.txt'), 'a+') as f:
                print(str(info.get('is_success', 0.0)), file=f)
        return observation, reward, done, info

class HERExplorer(explorer.Explorer):
    """Gaussian noise added to actions + Epsilon Greedy.

    Each action must be numpy.ndarray.

    Args:
        noise_std (float): percentage of action range that is std for noise.
        epsilon (float): Probability agent performs a rnd action.
        scale (float): Maximum action value.
    """

    def __init__(self, noise_std, epsilon, action_space):
        self.noise_std = noise_std
        self.epsilon = epsilon
        action_range = np.max(action_space.high) - np.min(action_space.low)
        self.std = noise_std * action_range
        self.action_space = action_space

    def select_action(self, t, greedy_action_func, action_value=None):
        if np.random.binomial(1, self.epsilon):
            a = self.random_action()
        else:
            a = greedy_action_func()
            noise = np.random.normal(
                scale=self.std, size=a.shape).astype(np.float32)
            a += noise
        return a

    def random_action(self):
        a = self.action_space.sample()
        if isinstance(a, np.ndarray):
            a = a.astype(np.float32)
        return a

    def __repr__(self):
        return 'AdditiveGaussian(noise_std={}, epsilon={})'.format(
            self.noise_std, self.epsilon)

def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='FetchPickAndPlace-v1')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 6)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--steps', type=int, default=200 * 50 * 16 * 50)
    parser.add_argument('--n-hidden-channels', type=int, default=64)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--replay-start-size', type=int, default=10000)
    parser.add_argument('--n-update-times', type=int, default=40)
    parser.add_argument('--target-update-interval',
                        type=int, default=16 * 50)
    parser.add_argument('--target-update-method',
                        type=str, default='soft', choices=['hard', 'soft'])
    parser.add_argument('--soft-update-tau', type=float, default=1 - 0.95)
    parser.add_argument('--update-interval', type=int, default=16 * 50)
    parser.add_argument('--eval-n-runs', type=int, default=30)
    parser.add_argument('--eval-interval', type=int, default=50 * 16 * 50)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--minibatch-size', type=int, default=128)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--noise-std', type=float, default=0.05)
    parser.add_argument('--clip-threshold', type=float, default=5.0)
    parser.add_argument('--num-envs', type=int, default=1)
    args = parser.parse_args()

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        if args.render and not test:
            env = chainerrl.wrappers.Render(env)
        if test:
            env = HEREnvWrapper(env, args.outdir)
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [(lambda: make_env(idx, test))
             for idx, env in enumerate(range(args.num_envs))])

    sample_env = make_env(0, test=False)

    def reward_function(state, action, goal):
        return sample_env.compute_reward(achieved_goal=state['achieved_goal'],
                                  desired_goal=goal,
                                  info=None)

    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    space_dict = sample_env.observation_space.spaces
    observation_space = space_dict['observation']
    goal_space = space_dict['desired_goal']
    obs_size = np.asarray(observation_space.shape).prod()
    goal_size = np.asarray(goal_space.shape).prod()
    action_space = sample_env.action_space

    action_size = np.asarray(action_space.shape).prod()    
    q_func = q_functions.FCSAQFunction(
        obs_size + goal_size, action_size,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=args.n_hidden_layers)
    pi = policy.FCDeterministicPolicy(
        obs_size + goal_size, action_size=action_size,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=args.n_hidden_layers,
        min_action=action_space.low, max_action=action_space.high,
        bound_action=True)
    model = DDPGModel(q_func=q_func, policy=pi)
    opt_a = optimizers.Adam(alpha=args.actor_lr)
    opt_c = optimizers.Adam(alpha=args.critic_lr)
    opt_a.setup(model['policy'])
    opt_c.setup(model['q_function'])
    opt_a.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
    opt_c.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')

    rbuf = replay_buffer.HindsightReplayBuffer(reward_function,
        10 ** 6,
        future_k=4)

    def phi(dict_state):
        return np.concatenate(
            (dict_state['observation'].astype(np.float32, copy=False),
            dict_state['desired_goal'].astype(np.float32, copy=False)), 0)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_size + goal_size, clip_threshold=args.clip_threshold)

    explorer = HERExplorer(args.noise_std,
        args.epsilon,
        action_space)
    agent = DDPG(model, opt_a, opt_c, rbuf,
                 obs_normalizer=obs_normalizer,
                 gamma=args.gamma,
                 explorer=explorer,
                 replay_start_size=args.replay_start_size,
                 phi=phi,
                 target_update_method=args.target_update_method,
                 target_update_interval=args.target_update_interval,
                 update_interval=args.update_interval,
                 soft_update_tau=args.soft_update_tau,
                 n_times_update=args.n_update_times,
                 gpu=args.gpu,
                 minibatch_size=args.minibatch_size,
                 clip_critic_tgt=(-1.0/(1.0-args.gamma), 0.0))

    if len(args.load) > 0:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent, env=make_batch_env(test=False), steps=args.steps,
            eval_env=make_batch_env(test=True), eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir,
            max_episode_len=timestep_limit)

if __name__ == '__main__':
    main()
