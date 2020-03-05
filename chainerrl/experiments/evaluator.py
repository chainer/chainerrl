import logging
import multiprocessing as mp
import os
import statistics
import time

import numpy as np

import chainerrl


"""Columns that describe information about an experiment.

steps: number of time steps taken (= number of actions taken)
episodes: number of episodes finished
elapsed: time elapsed so far (seconds)
mean: mean of returns of evaluation runs
median: median of returns of evaluation runs
stdev: stdev of returns of evaluation runs
max: maximum value of returns of evaluation runs
min: minimum value of returns of evaluation runs
"""
_basic_columns = ('steps', 'episodes', 'elapsed', 'mean',
                  'median', 'stdev', 'max', 'min')


def run_evaluation_episodes(env, agent, n_steps, n_episodes,
                            max_episode_len=None, logger=None):
    """Run multiple evaluation episodes and return returns.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    scores = []
    terminate = False
    timestep = 0

    reset = True
    while not terminate:
        if reset:
            obs = env.reset()
            done = False
            test_r = 0
            episode_len = 0
            info = {}
        a = agent.act(obs)
        obs, r, done, info = env.step(a)
        test_r += r
        episode_len += 1
        timestep += 1
        reset = (done or episode_len == max_episode_len
                 or info.get('needs_reset', False))
        if reset:
            logger.info('evaluation episode %s length:%s R:%s',
                        len(scores), episode_len, test_r)
            # As mixing float and numpy float causes errors in statistics
            # functions, here every score is cast to float.
            scores.append(float(test_r))
        if n_steps is None:
            terminate = len(scores) >= n_episodes
        else:
            terminate = timestep >= n_steps
        if reset or terminate:
            agent.stop_episode()
    # If all steps were used for a single unfinished episode
    if len(scores) == 0:
        scores.append(float(test_r))
        logger.info('evaluation episode %s length:%s R:%s',
                    len(scores), episode_len, test_r)
    return scores


def batch_run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns in a batch manner.

    Args:
        env (VectorEnv): Environment used for evaluation.
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of total timesteps to evaluate the agent.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes
            longer than this value will be truncated.
        logger (Logger or None): If specified, the given Logger
            object will be used for logging results. If not
            specified, the default logger of this module will
            be used.

    Returns:
        List of returns of evaluation runs.
    """
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_returns = dict()
    episode_lengths = dict()
    episode_indices = np.zeros(num_envs, dtype='i')
    episode_idx = 0
    for i in range(num_envs):
        episode_indices[i] = episode_idx
        episode_idx += 1
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_len = np.zeros(num_envs, dtype='i')

    obss = env.reset()
    rs = np.zeros(num_envs, dtype='f')

    termination_conditions = False
    timestep = 0
    while True:
        # a_t
        actions = agent.batch_act(obss)
        timestep += 1
        # o_{t+1}, r_{t+1}
        obss, rs, dones, infos = env.step(actions)
        episode_r += rs
        episode_len += 1
        # Compute mask for done and reset
        if max_episode_len is None:
            resets = np.zeros(num_envs, dtype=bool)
        else:
            resets = (episode_len == max_episode_len)
        resets = np.logical_or(
            resets, [info.get('needs_reset', False) for info in infos])

        # Make mask. 0 if done/reset, 1 if pass
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)

        for index in range(len(end)):
            if end[index]:
                episode_returns[episode_indices[index]] = episode_r[index]
                episode_lengths[episode_indices[index]] = episode_len[index]
                # Give the new episode an a new episode index
                episode_indices[index] = episode_idx
                episode_idx += 1

        episode_r[end] = 0
        episode_len[end] = 0

        # find first unfinished episode
        first_unfinished_episode = 0
        while first_unfinished_episode in episode_returns:
            first_unfinished_episode += 1

        # Check for termination conditions
        eval_episode_returns = []
        eval_episode_lens = []
        if n_steps is not None:
            total_time = 0
            for index in range(first_unfinished_episode):
                total_time += episode_lengths[index]
                # If you will run over allocated steps, quit
                if total_time > n_steps:
                    break
                else:
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
            termination_conditions = total_time >= n_steps
            if not termination_conditions:
                unfinished_index = np.where(
                    episode_indices == first_unfinished_episode)[0]
                if total_time + episode_len[unfinished_index] >= n_steps:
                    termination_conditions = True
                    if first_unfinished_episode == 0:
                        eval_episode_returns.append(
                            episode_r[unfinished_index])
                        eval_episode_lens.append(
                            episode_len[unfinished_index])

        else:
            termination_conditions = first_unfinished_episode >= n_episodes
            if termination_conditions:
                # Get the first n completed episodes
                for index in range(n_episodes):
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])

        if termination_conditions:
            # If this is the last step, make sure the agent observes reset=True
            resets.fill(True)

        # Agent observes the consequences.
        agent.batch_observe(obss, rs, dones, resets)

        if termination_conditions:
            break
        else:
            obss = env.reset(not_end)

    for i, (epi_len, epi_ret) in enumerate(
            zip(eval_episode_lens, eval_episode_returns)):
        logger.info('evaluation episode %s length: %s R: %s',
                    i, epi_len, epi_ret)
    return [float(r) for r in eval_episode_returns]


def eval_performance(env, agent, n_steps, n_episodes, max_episode_len=None,
                     logger=None):
    """Run multiple evaluation episodes and return statistics.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation episodes.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        Dict of statistics.
    """

    assert (n_steps is None) != (n_episodes is None)

    if isinstance(env, chainerrl.env.VectorEnv):
        scores = batch_run_evaluation_episodes(
            env, agent, n_steps, n_episodes,
            max_episode_len=max_episode_len,
            logger=logger)
    else:
        scores = run_evaluation_episodes(
            env, agent, n_steps, n_episodes,
            max_episode_len=max_episode_len,
            logger=logger)
    stats = dict(
        episodes=len(scores),
        mean=statistics.mean(scores),
        median=statistics.median(scores),
        stdev=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        max=np.max(scores),
        min=np.min(scores))
    return stats


def record_stats(outdir, values):
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        print('\t'.join(str(x) for x in values), file=f)


def save_agent(agent, t, outdir, logger, suffix=''):
    dirname = os.path.join(outdir, '{}{}'.format(t, suffix))
    agent.save(dirname)
    logger.info('Saved the agent to %s', dirname)


class Evaluator(object):
    """Object that is responsible for evaluating a given agent.

    Args:
        agent (Agent): Agent to evaluate.
        env (Env): Env to evaluate the agent on.
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean of returns in evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
    """

    def __init__(self,
                 agent,
                 env,
                 n_steps,
                 n_episodes,
                 eval_interval,
                 outdir,
                 max_episode_len=None,
                 step_offset=0,
                 save_best_so_far_agent=True,
                 logger=None,
                 ):
        assert (n_steps is None) != (n_episodes is None), \
            ("One of n_steps or n_episodes must be None. " +
             "Either we evaluate for a specified number " +
             "of episodes or for a specified number of timesteps.")
        self.agent = agent
        self.env = env
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.prev_eval_t = (self.step_offset -
                            self.step_offset % self.eval_interval)
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)

        # Write a header line first
        with open(os.path.join(self.outdir, 'scores.txt'), 'w') as f:
            custom_columns = tuple(t[0] for t in self.agent.get_statistics())
            column_names = _basic_columns + custom_columns
            print('\t'.join(column_names), file=f)

    def evaluate_and_update_max_score(self, t, episodes):
        eval_stats = eval_performance(
            self.env, self.agent, self.n_steps, self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger)
        elapsed = time.time() - self.start_time
        custom_values = tuple(tup[1] for tup in self.agent.get_statistics())
        mean = eval_stats['mean']
        values = (t,
                  episodes,
                  elapsed,
                  mean,
                  eval_stats['median'],
                  eval_stats['stdev'],
                  eval_stats['max'],
                  eval_stats['min']) + custom_values
        record_stats(self.outdir, values)
        if mean > self.max_score:
            self.logger.info('The best score is updated %s -> %s',
                             self.max_score, mean)
            self.max_score = mean
            if self.save_best_so_far_agent:
                save_agent(self.agent, "best", self.outdir, self.logger)
        return mean

    def evaluate_if_necessary(self, t, episodes):
        if t >= self.prev_eval_t + self.eval_interval:
            score = self.evaluate_and_update_max_score(t, episodes)
            self.prev_eval_t = t - t % self.eval_interval
            return score
        return None


class AsyncEvaluator(object):
    """Object that is responsible for evaluating asynchronous multiple agents.

    Args:
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
    """

    def __init__(self,
                 n_steps,
                 n_episodes,
                 eval_interval,
                 outdir,
                 max_episode_len=None,
                 step_offset=0,
                 save_best_so_far_agent=True,
                 logger=None,
                 ):
        assert (n_steps is None) != (n_episodes is None), \
            ("One of n_steps or n_episodes must be None. " +
             "Either we evaluate for a specified number " +
             "of episodes or for a specified number of timesteps.")
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)

        # Values below are shared among processes
        self.prev_eval_t = mp.Value(
            'l', self.step_offset - self.step_offset % self.eval_interval)
        self._max_score = mp.Value('f', np.finfo(np.float32).min)
        self.wrote_header = mp.Value('b', False)

        # Create scores.txt
        with open(os.path.join(self.outdir, 'scores.txt'), 'a'):
            pass

    @property
    def max_score(self):
        with self._max_score.get_lock():
            v = self._max_score.value
        return v

    def evaluate_and_update_max_score(self, t, episodes, env, agent):
        eval_stats = eval_performance(
            env, agent, self.n_steps, self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger)
        elapsed = time.time() - self.start_time
        custom_values = tuple(tup[1] for tup in agent.get_statistics())
        mean = eval_stats['mean']
        values = (t,
                  episodes,
                  elapsed,
                  mean,
                  eval_stats['median'],
                  eval_stats['stdev'],
                  eval_stats['max'],
                  eval_stats['min']) + custom_values
        record_stats(self.outdir, values)
        with self._max_score.get_lock():
            if mean > self._max_score.value:
                self.logger.info('The best score is updated %s -> %s',
                                 self._max_score.value, mean)
                self._max_score.value = mean
                if self.save_best_so_far_agent:
                    save_agent(agent, "best", self.outdir, self.logger)
        return mean

    def write_header(self, agent):
        with open(os.path.join(self.outdir, 'scores.txt'), 'w') as f:
            custom_columns = tuple(t[0] for t in agent.get_statistics())
            column_names = _basic_columns + custom_columns
            print('\t'.join(column_names), file=f)

    def evaluate_if_necessary(self, t, episodes, env, agent):
        necessary = False
        with self.prev_eval_t.get_lock():
            if t >= self.prev_eval_t.value + self.eval_interval:
                necessary = True
                self.prev_eval_t.value += self.eval_interval
        if necessary:
            with self.wrote_header.get_lock():
                if not self.wrote_header.value:
                    self.write_header(agent)
                    self.wrote_header.value = True
            return self.evaluate_and_update_max_score(t, episodes, env, agent)
        return None
