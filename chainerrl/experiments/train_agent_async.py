from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import range
from builtins import open
from builtins import str
from future import standard_library
standard_library.install_aliases()

import copy
import multiprocessing as mp
import os
import sys
import statistics
import time

import chainer
import numpy as np

from chainerrl.misc import random_seed
from chainerrl.misc import async


def eval_performance(process_idx, make_env, model, phi, n_runs, greedy=False, max_episode_len=None):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []

    for i in range(n_runs):
        model.reset_state()
        env = make_env(process_idx, test=True)
        if hasattr(env, 'spec'):
            timestep_limit = env.spec.timestep_limit
        else:
            timestep_limit = None
        obs = env.reset()
        done = False
        test_r = 0
        t = 0
        while not (done or t == max_episode_len):
            s = chainer.Variable(np.expand_dims(phi(obs), 0))
            pout, _ = model.pi_and_v(s)
            if greedy:
                a = pout.most_probable_actions.data[0]
            else:
                a = pout.sampled_actions.data[0]
            obs, r, done, info = env.step(a)
            test_r += r
            t += 1
            if timestep_limit is not None and t >= timestep_limit:
                break
        scores.append(test_r)
        print('test_{}:'.format(i), test_r)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


def train_loop(process_idx, counter, make_env, max_score, eval_frequency,
               eval_n_runs, agent, env, start_time, steps, outdir, max_episode_len=None):
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

        while True:

            total_r += r
            episode_r += r

            if done or episode_len == max_episode_len:
                agent.stop_episode_and_train(obs, r, done)
                if process_idx == 0:
                    print('{} global_t:{} local_t:{} lr:{} r:{}'.format(
                        outdir, global_t, local_t, agent.optimizer.lr,
                        episode_r))
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

                if global_t > steps:
                    break

                agent.optimizer.lr = (steps - global_t - 1) / steps * base_lr

                if global_t % eval_frequency == 0:
                    # Evaluation

                    # We must use a copy of the model because test runs can change
                    # the hidden states of the model
                    test_model = copy.deepcopy(agent.model)
                    test_model.reset_state()

                    mean, median, stdev = eval_performance(
                        process_idx, make_env, test_model, agent.phi,
                        eval_n_runs, max_episode_len=max_episode_len)
                    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
                        elapsed = time.time() - start_time
                        record = (global_t, elapsed, mean, median, stdev)
                        print('\t'.join(str(x) for x in record), file=f)
                    with max_score.get_lock():
                        if mean > max_score.value:
                            # Save the best model so far
                            print('The best score is updated {} -> {}'.format(
                                max_score.value, mean))
                            dirname = os.path.join(
                                outdir, '{}.h5'.format(global_t))
                            agent.save(dirname)
                            print('Saved the current best model to {}'.format(
                                dirname))
                            max_score.value = mean

    except KeyboardInterrupt:
        if process_idx == 0:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                outdir, '{}_keyboardinterrupt.h5'.format(global_t)))
            print('Saved the current model to {}'.format(
                outdir), file=sys.stderr)
        raise

    if global_t == steps + 1:
        # Save the final model
        dirname = os.path.join(outdir, '{}_finish'.format(steps))
        agent.save(dirname)
        print('Saved the final model to {}'.format(dirname))


def train_loop_with_profile(process_idx, counter, make_env, max_score,
                            eval_frequency, eval_n_runs, agent, env,
                            start_time, steps, outdir, max_episode_len=None):
    import cProfile
    cmd = 'train_loop(process_idx, counter, make_env, max_score, ' \
        'eval_frequency, eval_n_runs, agent, env, start_time, steps, ' \
        'outdir, max_episode_len)'
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
                      args={}):
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

    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    agent0 = make_agent(0)
    shared_objects = extract_shared_objects_from_agent(agent0)

    # Write a header line first
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    def run_func(process_idx):
        random_seed.set_random_seed(process_idx)

        env = make_env(process_idx, test=False)
        # agent = agent0 if process_idx == 0 else make_agent(process_idx)
        agent = make_agent(process_idx)
        set_shared_objects(agent, shared_objects)

        if profile:
            train_loop_with_profile(process_idx, counter, make_env, max_score,
                                    eval_frequency, eval_n_runs, agent, env,
                                    start_time, steps, outdir=outdir,
                                    max_episode_len=max_episode_len)
        else:
            train_loop(process_idx, counter, make_env, max_score,
                       eval_frequency, eval_n_runs, agent, env, start_time,
                       steps, outdir=outdir, max_episode_len=max_episode_len)

    async.run_async(processes, run_func)

    return agent0
