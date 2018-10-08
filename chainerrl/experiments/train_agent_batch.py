from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

from collections import deque
import logging
import os

import numpy as np


from chainerrl.experiments.evaluator import BatchEvaluator
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


def _get_mask(num_processes, info, done, episode_len, max_episode_len):
    if max_episode_len is None:
        reset_mask = np.zeros(num_processes, dtype=bool)
    else:
        reset_mask = episode_len == max_episode_len

    for i, reset in zip(info, reset_mask):
        i['reset'] = reset

    mask = np.logical_not(np.logical_or(reset_mask, done))
    return info, mask


def _write_to_file(outdir, step, avg_r):
    with open(os.path.join(outdir, 'training_r.txt'), 'a+') as f:
        print('\t'.join(str(x) for x in [step, avg_r]), file=f)


def train_agent_batch(agent, env, steps, outdir, log_interval=None,
                      max_episode_len=None, eval_interval=None,
                      step_offset=0, evaluator=None, successful_score=None,
                      step_hooks=[], return_window_size=100, logger=None,
                      save_training_r=False):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment train the againt against.
        steps (int): Number of total time steps for training.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size(int): Size of reward array.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See chainerrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
        save_training_r (bool): save training reward in outdir.
    """

    logger = logger or logging.getLogger(__name__)
    recent_returns = deque(maxlen=return_window_size)

    try:
        num_processes = env.num_envs
    except AttributeError:
        logger.error('Please pass a VectorEnv instance. \
You passed: {}'.format(type(env)))
        raise

    episode_r = np.zeros(num_processes, dtype='f')
    episode_idx = np.zeros(num_processes, dtype='i')
    episode_len = np.zeros(num_processes, dtype='i')

    # o_0, r_0
    obss = env.reset()
    rs = np.zeros(num_processes, dtype='f')

    t = step_offset
    if hasattr(agent, 't'):
        agent.t = step_offset

    try:
        while t < steps:
            # a_t
            actions = agent.batch_act_and_train(obss)
            # o_{t+1}, r_{t+1}
            obss, rs, dones, infos = env.step(actions)
            episode_r += rs
            episode_len += 1
            # Compute mask for done and reset
            if max_episode_len is None:
                end_by_episode_len = np.zeros(num_processes, dtype=bool)
            else:
                end_by_episode_len = (episode_len == max_episode_len)
            # Package mask inside info dict
            for info, flag in zip(infos, end_by_episode_len):
                info['reset'] = info.get('reset', False) or flag
            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(end_by_episode_len, dones)
            not_end = np.logical_not(end)
            # Reset environment whenever done/reset
            obss_ = env.reset(not_end)
            # Train agent
            agent.batch_observe_and_train(obss_, rs, dones, infos)
            # Update reward for current episode
            episode_idx += end
            # Add to deque whenever done/reset
            recent_returns.extend(
                np.ma.masked_array(episode_r, not_end).compressed())
            # Start new episode for those with mask
            episode_r *= not_end
            episode_len *= not_end
            t += num_processes

            for hook in step_hooks:
                hook(env, agent, t)

            if eval_interval is not None and t % log_interval == 0:
                logger.info('outdir:{}, step:{}, avg_r:{}, episode:{}'.format(
                    outdir, t, np.mean(recent_returns), np.sum(episode_idx)))
                logger.info('statistics: {}'.format(agent.get_statistics()))

                if evaluator is not None:
                    evaluator.evaluate_if_necessary(
                        t=t, episodes=episode_idx + 1)
                    if (successful_score is not None and
                            evaluator.max_score >= successful_score):
                        break

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix='_except')
        env.close()
        if evaluator:
            evaluator.env.close()
        raise
    else:
        # Save the final model
        save_agent(agent, t, outdir, logger, suffix='_finish')


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
                                      return_window_size=100,
                                      eval_env=None,
                                      log_interval=None,
                                      successful_score=None,
                                      step_hooks=[],
                                      save_best_so_far_agent=True,
                                      logger=None,
                                      save_training_r=False,
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
        return_window_size(int): Size of reward array.
        eval_explorer: Explorer used for evaluation.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If set to None, max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See chainerrl.experiments.hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        logger (logging.Logger): Logger used in this function.
        save_training_r (bool): save training reward in outdir.
    """

    logger = logger or logging.getLogger(__name__)

    makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = BatchEvaluator(agent=agent,
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
        eval_interval=eval_interval,
        evaluator=evaluator,
        successful_score=successful_score,
        return_window_size=return_window_size,
        log_interval=log_interval,
        step_hooks=step_hooks,
        save_training_r=save_training_r,
        logger=logger)
