import logging
import multiprocessing as mp
import os

import numpy as np

from chainerrl.experiments.evaluator import AsyncEvaluator
from chainerrl.misc import async_
from chainerrl.misc import random_seed


def train_loop(process_idx, env, agent, steps, outdir, counter,
               episodes_counter, stop_event,
               max_episode_len=None, evaluator=None, eval_env=None,
               successful_score=None, logger=None,
               global_step_hooks=()):

    logger = logger or logging.getLogger(__name__)

    if eval_env is None:
        eval_env = env

    try:

        episode_r = 0
        global_t = 0
        local_t = 0
        global_episodes = 0
        obs = env.reset()
        r = 0
        done = False
        episode_len = 0
        successful = False

        while True:

            # a_t
            a = agent.act_and_train(obs, r)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(a)
            local_t += 1
            episode_r += r
            episode_len += 1

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value

            for hook in global_step_hooks:
                hook(env, agent, global_t)

            reset = (episode_len == max_episode_len
                     or info.get('needs_reset', False))
            if done or reset or global_t >= steps or stop_event.is_set():
                agent.stop_episode_and_train(obs, r, done)

                if process_idx == 0:
                    logger.info(
                        'outdir:%s global_step:%s local_step:%s R:%s',
                        outdir, global_t, local_t, episode_r)
                    logger.info('statistics:%s', agent.get_statistics())

                # Evaluate the current agent
                if evaluator is not None:
                    eval_score = evaluator.evaluate_if_necessary(
                        t=global_t, episodes=global_episodes,
                        env=eval_env, agent=agent)
                    if (eval_score is not None and
                            successful_score is not None and
                            eval_score >= successful_score):
                        stop_event.set()
                        successful = True
                        # Break immediately in order to avoid an additional
                        # call of agent.act_and_train
                        break

                with episodes_counter.get_lock():
                    episodes_counter.value += 1
                    global_episodes = episodes_counter.value

                if global_t >= steps or stop_event.is_set():
                    break

                # Start a new episode
                episode_r = 0
                episode_len = 0
                obs = env.reset()
                r = 0

    except (Exception, KeyboardInterrupt):
        if process_idx == 0:
            # Save the current model before being killed
            dirname = os.path.join(outdir, '{}_except'.format(global_t))
            agent.save(dirname)
            logger.warning('Saved the current model to %s', dirname)
        raise

    if global_t == steps:
        # Save the final model
        dirname = os.path.join(outdir, '{}_finish'.format(steps))
        agent.save(dirname)
        logger.info('Saved the final agent to %s', dirname)

    if successful:
        # Save the successful model
        dirname = os.path.join(outdir, 'successful')
        agent.save(dirname)
        logger.info('Saved the successful agent to %s', dirname)


def extract_shared_objects_from_agent(agent):
    return dict((attr, async_.as_shared_objects(getattr(agent, attr)))
                for attr in agent.shared_attributes)


def set_shared_objects(agent, shared_objects):
    for attr, shared in shared_objects.items():
        new_value = async_.synchronize_to_shared_objects(
            getattr(agent, attr), shared)
        setattr(agent, attr, new_value)


def train_agent_async(
    outdir,
    processes,
    make_env,
    profile=False,
    steps=8 * 10 ** 7,
    eval_interval=10 ** 6,
    eval_n_steps=None,
    eval_n_episodes=10,
    max_episode_len=None,
    step_offset=0,
    successful_score=None,
    agent=None,
    make_agent=None,
    global_step_hooks=[],
    save_best_so_far_agent=True,
    logger=None,
    random_seeds=None,
    stop_event=None,
):
    """Train agent asynchronously using multiprocessing.

    Either `agent` or `make_agent` must be specified.

    Args:
        outdir (str): Path to the directory to output things.
        processes (int): Number of processes.
        make_env (callable): (process_idx, test) -> Environment.
        profile (bool): Profile if set True.
        steps (int): Number of global time steps for training.
        eval_interval (int): Interval of evaluation. If set to None, the agent
            will not be evaluated at all.
        eval_n_steps (int): Number of eval timesteps at each eval phase
        eval_n_episodes (int): Number of eval episodes at each eval phase
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        successful_score (float): Finish training if the mean score is greater
            or equal to this value if not None
        agent (Agent): Agent to train.
        make_agent (callable): (process_idx) -> Agent
        global_step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every global
            step. See chainerrl.experiments.hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        logger (logging.Logger): Logger used in this function.
        random_seeds (array-like of ints or None): Random seeds for processes.
            If set to None, [0, 1, ..., processes-1] are used.
        stop_event (multiprocessing.Event or None): Event to stop training.
            If set to None, a new Event object is created and used internally.

    Returns:
        Trained agent.
    """

    logger = logger or logging.getLogger(__name__)

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    counter = mp.Value('l', 0)
    episodes_counter = mp.Value('l', 0)

    if stop_event is None:
        stop_event = mp.Event()

    if agent is None:
        assert make_agent is not None
        agent = make_agent(0)

    shared_objects = extract_shared_objects_from_agent(agent)
    set_shared_objects(agent, shared_objects)

    if eval_interval is None:
        evaluator = None
    else:
        evaluator = AsyncEvaluator(
            n_steps=eval_n_steps,
            n_episodes=eval_n_episodes,
            eval_interval=eval_interval, outdir=outdir,
            max_episode_len=max_episode_len,
            step_offset=step_offset,
            save_best_so_far_agent=save_best_so_far_agent,
            logger=logger,
        )

    if random_seeds is None:
        random_seeds = np.arange(processes)

    def run_func(process_idx):
        random_seed.set_random_seed(random_seeds[process_idx])

        env = make_env(process_idx, test=False)
        if evaluator is None:
            eval_env = env
        else:
            eval_env = make_env(process_idx, test=True)
        if make_agent is not None:
            local_agent = make_agent(process_idx)
            set_shared_objects(local_agent, shared_objects)
        else:
            local_agent = agent
        local_agent.process_idx = process_idx

        def f():
            train_loop(
                process_idx=process_idx,
                counter=counter,
                episodes_counter=episodes_counter,
                agent=local_agent,
                env=env,
                steps=steps,
                outdir=outdir,
                max_episode_len=max_episode_len,
                evaluator=evaluator,
                successful_score=successful_score,
                stop_event=stop_event,
                eval_env=eval_env,
                global_step_hooks=global_step_hooks,
                logger=logger)

        if profile:
            import cProfile
            cProfile.runctx('f()', globals(), locals(),
                            'profile-{}.out'.format(os.getpid()))
        else:
            f()

        env.close()
        if eval_env is not env:
            eval_env.close()

    async_.run_async(processes, run_func)

    stop_event.set()

    return agent
