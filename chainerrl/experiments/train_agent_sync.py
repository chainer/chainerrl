from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import logging

import numpy as np

from chainerrl.experiments.evaluator import Evaluator
from chainerrl.experiments.evaluator import save_agent
from chainerrl.misc.makedirs import makedirs


def train_agent_sync(agent, env, steps, outdir, log_interval=None,
                     eval_interval=None, step_offset=0, evaluator=None,
                     successful_score=None, step_hooks=[], logger=None):

    logger = logger or logging.getLogger(__name__)

    obs = env.reset()
    t = step_offset
    if hasattr(agent, 't'):
        agent.t = step_offset

    num_processes = env.num_envs
    episode_r = np.zeros(num_processes, dtype='f')
    episode_final_r = np.zeros(num_processes, dtype='f')
    r = np.zeros((num_processes, 1), dtype='f')
    done = np.zeros((num_processes, 1), dtype=np.bool)
    episode_idx = 0

    try:
        while t < steps:
            action = agent.act_and_train(obs, r, done)
            obs, r, done, info = env.step(action)
            masks = np.array([0.0 if done_ else 1.0 for done_ in done])
            episode_r += r
            episode_final_r *= masks
            episode_final_r += (1 - masks) * episode_r
            episode_r *= masks

            r = np.expand_dims(np.stack(r), 1)
            t += 1

            for hook in step_hooks:
                hook(env, agent, t)

            if eval_interval is not None and t % log_interval == 0:
                logger.info('outdir:%s step:%s \
episode:%s mean/median \
reward:%s/%s min/max reward:%s/%s',
                            outdir, t, t * agent.update_steps,
                            np.mean(episode_final_r),
                            np.median(episode_final_r),
                            np.min(episode_final_r),
                            np.max(episode_final_r))
                logger.info('statistics:%s', agent.get_statistics())

            if eval_interval is not None and t % eval_interval == 0:
                if evaluator is not None:
                    evaluator.evaluate_if_necessary(
                        t=t, episodes=episode_idx + 1)
                    episode_idx += 1
                    if (successful_score is not None and
                            evaluator.max_score >= successful_score):
                        break

    except Exception:
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix='_except')
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix='_finish')


def train_agent_with_evaluation_sync(
        agent, env, steps, eval_n_runs, eval_interval,
        log_interval, outdir, step_offset=0, eval_explorer=None,
        eval_max_episode_len=None, eval_env=None, successful_score=None,
        step_hooks=[], logger=None):
    """Train an agent while regularly evaluating it.

    Args:
        agent: Agent to train.
        env: Environment train the agent against.
        steps (int): Number of total time steps for training.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        log_interval (int): Interval of log output.
        outdir (str): Path to the directory to output things.
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

    evaluator = Evaluator(agent=agent,
                          n_runs=eval_n_runs,
                          eval_interval=eval_interval,
                          outdir=outdir,
                          max_episode_len=eval_max_episode_len,
                          explorer=eval_explorer,
                          env=eval_env,
                          step_offset=step_offset,
                          logger=logger)

    train_agent_sync(
        agent, env, steps, outdir,
        log_interval=log_interval,
        eval_interval=eval_interval,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        logger=logger)
