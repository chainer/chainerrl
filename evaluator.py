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

import numpy as np


def eval_performance(env, agent, n_runs, max_episode_len=None,
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


def record_stats(outdir, values):
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        print('\t'.join(str(x) for x in values), file=f)


def save_agent_model(agent, t, outdir, suffix=''):
    filename = os.path.join(outdir, '{}{}.h5'.format(t, suffix))
    agent.save_model(filename)
    print('Saved the current model to {}'.format(filename))


def update_best_model(agent, outdir, t, old_max_score, new_max_score):
    # Save the best model so far
    print('The best score is updated {} -> {}'.format(
        old_max_score, new_max_score))
    save_agent_model(agent, t, outdir)


class Evaluator(object):

    def __init__(self, agent, env, n_runs, eval_frequency,
                 outdir, max_episode_len=None, explorer=None,
                 step_offset=0):
        self.agent = agent
        self.env = env
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.eval_after_this_episode = False
        self.n_runs = n_runs
        self.eval_frequency = eval_frequency
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.explorer = explorer
        self.step_offset = step_offset
        self.prev_eval_t = self.step_offset - self.step_offset % self.eval_frequency

        # Write a header line first
        with open(os.path.join(self.outdir, 'scores.txt'), 'a+') as f:
            column_names = (('steps', 'elapsed', 'mean', 'median', 'stdev') +
                            self.agent.get_stats_keys())
            print('\t'.join(column_names), file=f)

    def evaluate_and_update_max_score(self, t):
        mean, median, stdev = eval_performance(
            self.env, self.agent, self.n_runs,
            max_episode_len=self.max_episode_len, explorer=self.explorer)
        elapsed = time.time() - self.start_time
        values = (t, elapsed, mean, median, stdev) + \
            self.agent.get_stats_values()
        record_stats(self.outdir, values)
        if mean > self.max_score:
            update_best_model(self.agent, self.outdir, t, self.max_score, mean)
            self.max_score = mean

    def evaluate_if_necessary(self, t):
        if t >= self.prev_eval_t + self.eval_frequency:
            self.evaluate_and_update_max_score(t)
            self.prev_eval_t = t - t % self.eval_frequency
