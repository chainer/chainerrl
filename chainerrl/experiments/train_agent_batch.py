from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import logging
import os

import numpy as np


from chainerrl.experiments.evaluator import Evaluator
from chainerrl.experiments.evaluator import save_agent
from chainerrl.misc.ask_yes_no import ask_yes_no
from chainerrl.misc.makedirs import makedirs


def save_agent_replay_buffer(agent, t, outdir, suffix='', logger=None):
    logger = logger or logging.getLogger(__name__)
    filename = os.path.join(outdir, '{}{}.replay.pkl'.format(t, suffix))
    agent.replay_buffer.save(filename)
    logger.info('Saved the current replay buffer to %s', filename)


def ask_and_save_agent_replay_buffer(agent, t, outdir, suffix=''):
    if hasattr(agent, 'replay_buffer') and \
            ask_yes_no('Replay buffer has {} transitions. Do you save them to a file?'.format(len(agent.replay_buffer))):  # NOQA
        save_agent_replay_buffer(agent, t, outdir, suffix=suffix)


def train_agent_batch(agent, env, steps, outdir, max_episode_len=None,
                      step_offset=0, evaluator=None, successful_score=None,
                      step_hooks=[], logger=None):

    logger = logger or logging.getLogger(__name__)

    try:
        num_processes = env.num_envs
    except AttributeError:
        logger.error('Please pass a VectorEnv instance. \
You passed: {}'.format(type(env)))
        raise

    episode_r = np.zeros(num_processes, dtype='f')
    episode_idx = np.zeros(num_processes, dtype='f')

    # o_0, r_0
    obs = env.reset()
    r = np.zeros(num_processes, dtype='f')

    t = step_offset
    if hasattr(agent, 't'):
        agent.t = step_offset

    try:
        while t < steps:
            # a_t : Make an action
            action = agent.batch_act(obs)
            # o_{t+1}, r_{t+1} : Get observation and reward
            obs, r, done, info = env.step(action)
            logger.info('outdir:{}, step:{} episode:{}, R:{}'.format(
                outdir, t, episode_idx, episode_r))
            logger.info('statistics: {}'.format(agent.get_statistics()))
            # Train model
            if max_episode_len is None:
                reset_mask = np.zeros(num_processes, dtype=bool)
            else:
                reset_mask = episode_idx == max_episode_len
            for i, reset in zip(info, reset_mask):
                i['reset'] = reset
            agent.batch_observe(obs, r, done, info)
            # Update counters
            t += 1
            # Update episode reward
            episode_r += r
            # Reset episode reward whenever done or reset signal
            # is given
            mask = np.logical_or(reset_mask, done)
            episode_r_ = np.ma.array(episode_r, mask=mask)
            episode_idx[episode_r_.mask] += 1
            episode_r = episode_r_.filled(0)
    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix='_except')
        env.close()
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix='_finish')
    env.close()


def train_agent_batch_with_evaluation(agent,
                                      env,
                                      steps,
                                      eval_n_runs,
                                      eval_interval,
                                      outdir,
                                      max_episode_len=None,
                                      step_offset=0,
                                      eval_explorer=None,
                                      eval_max_episode_len=None,
                                      eval_env=None,
                                      successful_score=None,
                                      step_hooks=[],
                                      save_best_so_far_agent=True,
                                      logger=None,
                                      ):
    """Train an agent while regularly evaluating it.

    Args:
        agent: Agent to train.
        env: Environment train the againt against.
        steps (int): Number of total time steps for training.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        eval_explorer: Explorer used for evaluation.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If set to None, max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            or equal to this value if not None
        step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See chainerrl.experiments.hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)

    makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = Evaluator(agent=agent,
                          n_runs=eval_n_runs,
                          eval_interval=eval_interval, outdir=outdir,
                          max_episode_len=eval_max_episode_len,
                          explorer=eval_explorer,
                          env=eval_env,
                          step_offset=step_offset,
                          save_best_so_far_agent=save_best_so_far_agent,
                          logger=logger,
                          )

    train_agent_batch(
        agent, env, steps, outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        logger=logger)
