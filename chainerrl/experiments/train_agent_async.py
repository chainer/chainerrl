from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *
from future import standard_library
standard_library.install_aliases()

import multiprocessing as mp
import os
import sys

from chainerrl.misc import random_seed
from chainerrl.misc import async
from chainerrl.experiments.evaluator import AsyncEvaluator


def train_loop(process_idx, env, agent, steps, outdir, counter, training_done,
               max_episode_len=None, evaluator=None, eval_env=None,
               successful_score=None):

    if eval_env is None:
        eval_env = env

    try:

        total_r = 0
        episode_r = 0
        global_t = 0
        local_t = 0
        obs = env.reset()
        r = 0
        done = False
        base_lr = agent.optimizer.lr
        episode_len = 0
        successful = False

        while True:

            total_r += r
            episode_r += r

            if done or episode_len == max_episode_len:
                agent.stop_episode_and_train(obs, r, done)
                if process_idx == 0:
                    print('{} global_t:{} local_t:{} lr:{} r:{}'.format(
                        outdir, global_t, local_t, agent.optimizer.lr,
                        episode_r))
                if evaluator is not None:
                    eval_score = evaluator.evaluate_if_necessary(
                        global_t, env=eval_env, agent=agent)
                    if (eval_score is not None and
                            successful_score is not None and
                            eval_score >= successful_score):
                        with training_done.get_lock():
                            if not training_done.value:
                                training_done.value = True
                                successful = True
                        # Break immediately in order to avoid an additional
                        # call of agent.act_and_train
                        break
                episode_r = 0
                obs = env.reset()
                r = 0
                done = False
                episode_len = 0
            else:
                a = agent.act_and_train(obs, r)
                obs, r, done, info = env.step(a)

                # Get and increment the global counter
                with counter.get_lock():
                    counter.value += 1
                    global_t = counter.value
                local_t += 1
                episode_len += 1

                if global_t > steps or training_done.value:
                    break

                agent.optimizer.lr = (steps - global_t - 1) / steps * base_lr

    except KeyboardInterrupt:
        if process_idx == 0:
            # Save the current model before being killed
            dirname = os.path.join(outdir, '{}_except'.format(global_t))
            agent.save(dirname)
            print('Saved the current model to {}'.format(dirname),
                  file=sys.stderr)
        raise

    if global_t == steps + 1:
        # Save the final model
        dirname = os.path.join(outdir, '{}_finish'.format(steps))
        agent.save(dirname)
        print('Saved the final model to {}'.format(dirname))

    if successful:
        # Save the successful model
        dirname = os.path.join(outdir, 'successful')
        agent.save(dirname)
        print('Saved the successful model to {}'.format(dirname))


def train_loop_with_profile(process_idx, counter, agent, env,
                            start_time, steps, outdir, max_episode_len=None,
                            evaluator=None, eval_env=None):
    import cProfile
    cmd = 'train_loop(process_idx=process_idx, counter=counter, agent=agent, env=env, steps=steps, outdir=outdir, max_episode_len=max_episode_len, evaluator=evaluator)'
    cProfile.runctx(cmd, globals(), locals(),
                    'profile-{}.out'.format(os.getpid()))


def extract_shared_objects_from_agent(agent):
    return dict((attr, async.as_shared_objects(getattr(agent, attr)))
                for attr in agent.shared_attributes)


def set_shared_objects(agent, shared_objects):
    for attr, shared in shared_objects.items():
        new_value = async.synchronize_to_shared_objects(
            getattr(agent, attr), shared)
        setattr(agent, attr, new_value)


def train_agent_async(outdir, processes, make_env, make_agent,
                      profile=False, steps=8 * 10 ** 7, eval_frequency=10 ** 6,
                      eval_n_runs=10, gamma=0.99, max_episode_len=None,
                      step_offset=0, successful_score=None):
    """
    Args:
      processes (int): Number of processes.
      make_env (callable): (process_idx, test) -> env
      model_opt (callable): () -> (models, optimizers)
      make_agent (callable): (process_idx, models, optimizers) -> agent
      profile (bool): Profile if set True
      steps (int): Number of global time steps for training
    """

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    counter = mp.Value('l', 0)
    training_done = mp.Value('b', False)  # bool

    agent0 = make_agent(0)
    shared_objects = extract_shared_objects_from_agent(agent0)
    set_shared_objects(agent0, shared_objects)

    evaluator = AsyncEvaluator(
        n_runs=eval_n_runs,
        eval_frequency=eval_frequency, outdir=outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset)

    def run_func(process_idx):
        random_seed.set_random_seed(process_idx)

        env = make_env(process_idx, test=False)
        # agent = agent0 if process_idx == 0 else make_agent(process_idx)
        agent = make_agent(process_idx)
        set_shared_objects(agent, shared_objects)

        if profile:
            train_loop_with_profile(
                process_idx=process_idx,
                counter=counter,
                agent=agent,
                env=env,
                steps=steps,
                outdir=outdir,
                max_episode_len=max_episode_len,
                evaluator=evaluator,
                successful_score=successful_score,
                training_done=training_done)
        else:
            train_loop(
                process_idx=process_idx,
                counter=counter,
                agent=agent,
                env=env,
                steps=steps,
                outdir=outdir,
                max_episode_len=max_episode_len,
                evaluator=evaluator,
                successful_score=successful_score,
                training_done=training_done)

    async.run_async(processes, run_func)

    return agent0
