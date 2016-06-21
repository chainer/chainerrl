import os
import statistics
import time

import chainer
import numpy as np


def eval_performance(make_env, q_func, phi, n_runs, gpu):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []
    for i in range(n_runs):
        env = make_env()
        obs = env.reset()
        done = False
        test_r = 0
        while not done:
            s = np.expand_dims(phi(obs), 0)
            if gpu >= 0:
                s = chainer.cuda.to_gpu(s)
            qout = q_func(chainer.Variable(s), test=True)
            a = qout.greedy_actions.data[0]
            obs, r, done, info = env.step(a)
            test_r += r
        scores.append(test_r)
        print('test_{}:'.format(i), test_r)
        env.close()
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


def run_dqn(agent, make_env, phi, steps, eval_n_runs, eval_frequency, gpu,
            outdir):

    env = make_env()

    episode_r = 0

    episode_idx = 0
    max_score = np.finfo(np.float32).min

    # Write a header line first
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    start_time = time.time()

    obs = env.reset()
    r = 0
    done = False

    t = 0
    while t < steps:
        try:
            if t % eval_frequency == 0:
                # Test performance
                mean, median, stdev = eval_performance(
                    make_env, agent.q_function, phi, eval_n_runs, gpu)
                with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
                    elapsed = time.time() - start_time
                    record = (t, elapsed, mean, median, stdev)
                    print('\t'.join(str(x) for x in record), file=f)
                if mean > max_score:
                    if max_score is not None:
                        # Save the best model so far
                        print('The best score is updated {} -> {}'.format(
                            max_score, mean))
                        filename = os.path.join(outdir, '{}.h5'.format(t))
                        agent.save_model(filename)
                        print('Saved the current best model to {}'.format(
                            filename))
                    max_score = mean

            episode_r += r

            action = agent.act(obs, r, done)

            if done:
                print('{} t:{} episode_idx:{} explorer:{} episode_r:{}'.format(
                    outdir, t, episode_idx, agent.explorer, episode_r))
                episode_r = 0
                episode_idx += 1
                obs = env.reset()
                r = 0
                done = False
            else:
                obs, r, done, info = env.step(action)
                t += 1
        except KeyboardInterrupt:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                outdir, '{}_keyboardinterrupt.h5'.format(t)))
            print('Saved the current model to {}'.format(outdir))
            raise

    # Save the final model
    agent.save_model(os.path.join(
        outdir, '{}_finish.h5'.format(steps)))
    print('Saved the current model to {}'.format(outdir))
