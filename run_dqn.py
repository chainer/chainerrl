from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import os

from agents.dqn import DQN
from ask_yes_no import ask_yes_no
from evaluator import Evaluator


def save_agent_model(agent, t, outdir, suffix=''):
    filename = os.path.join(outdir, '{}{}.h5'.format(t, suffix))
    agent.save_model(filename)
    print('Saved the current model to {}'.format(filename))


def save_agent_replay_buffer(agent, t, outdir, suffix=''):
    filename = os.path.join(outdir, '{}{}.replay.pkl'.format(t, suffix))
    agent.replay_buffer.save(filename)
    print('Saved the current replay buffer to {}'.format(filename))


def ask_and_save_agent_replay_buffer(agent, t, outdir, suffix=''):
    if hasattr(agent, 'replay_buffer') and \
            ask_yes_no('Replay buffer has {} transitions. Do you save them to a file?'.format(len(agent.replay_buffer))):
        save_agent_replay_buffer(agent, t, outdir, suffix=suffix)


def run_dqn_with_evaluation(agent, env, steps, outdir, max_episode_len=None,
                            step_offset=0, evaluator=None,
                            save_final_model=True):

    episode_r = 0
    episode_idx = 0

    obs = env.reset()
    r = 0
    done = False

    t = step_offset
    agent.t = step_offset

    episode_len = 0
    while t < steps:
        try:
            episode_r += r

            if done or episode_len == max_episode_len:
                if done:
                    agent.observe_terminal(obs, r)
                else:
                    agent.stop_current_episode(obs, r)
                print('{} t:{} episode_idx:{} explorer:{} episode_r:{}'.format(
                    outdir, t, episode_idx, agent.explorer, episode_r))
                if evaluator is not None:
                    evaluator.evaluate_if_necessary(t)
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
            save_agent_model(agent, t, outdir, suffix='_except')
            ask_and_save_agent_replay_buffer(
                agent, t, outdir, suffix='_except')
            raise

    if save_final_model:
        # Save the final model
        save_agent_model(agent, t, outdir, suffix='_finish')
        ask_and_save_agent_replay_buffer(agent, t, outdir, suffix='_finish')


def run_dqn(agent, env, steps, eval_n_runs, eval_frequency,
            outdir, max_episode_len=None, step_offset=0, eval_explorer=None,
            eval_env=None):
    """
    Run a DQN-like agent.

    Args:
      agent: Agent.
      env: Environment.
      steps (int): Number of total time steps for training.
      eval_n_runs (int): Number of runs for each time of evaluation.
      eval_frequency (int): Interval of evaluation.
      outdir (str): Path to the directory to output things.
      max_episode_len (int): Maximum episode length.
      step_offset (int): Time step from which training starts.
      eval_explorer: Explorer used for evaluation.
      eval_env: Environment used for evaluation.
    """

    if eval_env is None:
        eval_env = env

    evaluator = Evaluator(agent=agent,
                          n_runs=eval_n_runs,
                          eval_frequency=eval_frequency, outdir=outdir,
                          max_episode_len=max_episode_len,
                          explorer=eval_explorer,
                          env=eval_env,
                          step_offset=step_offset)

    run_dqn_with_evaluation(
        agent, env, steps, outdir, max_episode_len=max_episode_len,
        step_offset=step_offset, evaluator=evaluator)
