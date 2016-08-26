from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import open
from builtins import str
from future import standard_library
standard_library.install_aliases()
import os
import statistics
import time

import chainer
import numpy as np

from agents.dqn import DQN


def eval_performance(env, agent, n_runs, gpu, max_episode_len=None,
                     explorer=None):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []
    for i in range(n_runs):
        obs = env.reset()
        done = False
        test_r = 0
        t = 0
        while not (done or t == max_episode_len):
            def greedy_action_func():
                return agent.select_greedy_action(obs)
            if explorer is not None:
                a = explorer.select_action(t, greedy_action_func)
            else:
                a = greedy_action_func()
            obs, r, done, info = env.step(a)
            test_r += r
            t += 1
        # As mixing float and numpy float causes errors in statistics
        # functions, here every score is cast to float.
        scores.append(float(test_r))
        print('test_{}:'.format(i), test_r)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


def record_stats(outdir, t, start_time, mean, median, stdev):
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        elapsed = time.time() - start_time
        record = (t, elapsed, mean, median, stdev)
        print('\t'.join(str(x) for x in record), file=f)


def update_best_model(agent, outdir, t, old_max_score, new_max_score):
    # Save the best model so far
    print('The best score is updated {} -> {}'.format(
        old_max_score, new_max_score))
    filename = os.path.join(outdir, '{}.h5'.format(t))
    agent.save_model(filename)
    print('Saved the current best model to {}'.format(
        filename))


class Evaluator(object):

    def __init__(self, n_runs, gpu, eval_frequency,
                 outdir, max_episode_len=None, explorer=None):
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.eval_after_this_episode = False
        self.n_runs = n_runs
        self.gpu = gpu
        self.eval_frequency = eval_frequency
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.explorer = explorer
        self.prev_eval_t = 0

    def evaluate_and_update_max_score(self, env, t, agent):
        mean, median, stdev = eval_performance(
            env, agent, self.n_runs, self.gpu,
            max_episode_len=self.max_episode_len, explorer=self.explorer)
        record_stats(self.outdir, t, self.start_time, mean, median, stdev)
        if mean > self.max_score:
            update_best_model(agent, self.outdir, t, self.max_score, mean)
            self.max_score = mean

    def evaluate_if_necessary(self, t, env, agent):
        if t >= self.prev_eval_t + self.eval_frequency:
            self.evaluate_and_update_max_score(env, t, agent)
            self.prev_eval_t = t


def run_dqn(agent, make_env, phi, steps, eval_n_runs, eval_frequency, gpu,
            outdir, max_episode_len=None, step_offset=0, eval_explorer=None):

    assert isinstance(agent, DQN)

    env = make_env(False)

    episode_r = 0

    episode_idx = 0

    # Write a header line first
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    obs = env.reset()
    r = 0
    done = False

    t = step_offset
    agent.t = step_offset

    evaluator = Evaluator(n_runs=eval_n_runs, gpu=gpu,
                          eval_frequency=eval_frequency, outdir=outdir,
                          max_episode_len=max_episode_len,
                          explorer=eval_explorer)
    evaluator.prev_eval_t = step_offset - step_offset % eval_frequency

    episode_len = 0
    while t < steps:
        try:
            episode_r += r

            if done or episode_len == max_episode_len:
                if done:
                    agent.observe_terminal(obs, r)
                else:
                    agent.stop_current_episode()
                print('{} t:{} episode_idx:{} explorer:{} episode_r:{}'.format(
                    outdir, t, episode_idx, agent.explorer, episode_r))
                evaluator.evaluate_if_necessary(t, env, agent)
                # Start a new episode
                episode_r = 0
                episode_idx += 1
                episode_len = 0
                obs = env.reset()
                r = 0
                done = False
            else:
                action = agent.act(obs, r)
                obs, r, done, info = env.step(action)
                t += 1
                episode_len += 1

        except:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                outdir, '{}_keyboardinterrupt.h5'.format(t)))
            print('Saved the current model to {}'.format(outdir))
            raise

    # Save the final model
    agent.save_model(os.path.join(
        outdir, '{}_finish.h5'.format(steps)))
    print('Saved the current model to {}'.format(outdir))
