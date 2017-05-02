from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import logging
import os

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


def train_agent(agent, env, steps, outdir, max_episode_len=None,
                step_offset=0, evaluator=None, successful_score=None,
                step_hooks=[], logger=None):

    logger = logger or logging.getLogger(__name__)

    episode_r = 0
    episode_idx = 0

    # o_0, r_0
    obs = env.reset()
    r = 0
    done = False

    t = step_offset
    agent.t = step_offset

    episode_len = 0
    try:
        while t < steps:

            # a_t
            action = agent.act_and_train(obs, r)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(action)
            t += 1
            episode_r += r
            episode_len += 1

            for hook in step_hooks:
                hook(env, agent, t)

            if done or episode_len == max_episode_len or t == steps:
                agent.stop_episode_and_train(obs, r, done=done)
                logger.info('outdir:%s step:%s episode:%s R:%s',
                            outdir, t, episode_idx, episode_r)
                logger.info('statistics:%s', agent.get_statistics())
                if evaluator is not None:
                    evaluator.evaluate_if_necessary(
                        t=t, episodes=episode_idx + 1)
                    if (successful_score is not None and
                            evaluator.max_score >= successful_score):
                        break
                if t == steps:
                    break
                # Start a new episode
                episode_r = 0
                episode_idx += 1
                episode_len = 0
                obs = env.reset()
                r = 0
                done = False

    except Exception:
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix='_except')
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix='_finish')


def train_agent_with_evaluation(
        agent, env, steps, eval_n_runs, eval_interval,
        outdir, max_episode_len=None, step_offset=0, eval_explorer=None,
        eval_max_episode_len=None, eval_env=None, successful_score=None,
        step_hooks=[], logger=None):
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
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            or equal to this value if not None
        step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See chainerrl.experiments.hooks.
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
                          logger=logger)

    train_agent(
        agent, env, steps, outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        logger=logger)
