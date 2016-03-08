import multiprocessing as mp
import os
import random

import numpy as np

import random_seed


def set_shared_params(a, b):
    for param_name, param in a.namedparams():
        if param_name in b:
            shared_param = b[param_name]
            param.data = np.frombuffer(shared_param.get_obj(
            ), dtype=param.data.dtype).reshape(param.data.shape)


def set_shared_states(a, b):
    for state_name, shared_state in b.iteritems():
        for param_name, param in shared_state.iteritems():
            old_param = a._states[state_name][param_name]
            a._states[state_name][param_name] = np.frombuffer(
                param.get_obj(),
                dtype=old_param.dtype).reshape(old_param.shape)


def extract_params_as_shared_arrays(link):
    shared_arrays = {}
    for param_name, param in link.namedparams():
        shared_arrays[param_name] = mp.Array('f', param.data.ravel())
    return shared_arrays


def extract_states_as_shared_arrays(optimizer):
    shared_arrays = {}
    for state_name, state in optimizer._states.iteritems():
        shared_arrays[state_name] = {}
        for param_name, param in state.iteritems():
            shared_arrays[state_name][
                param_name] = mp.Array('f', param.ravel())
    return shared_arrays


def run_a3c_process(agent_func, env_func, run_func, link_arrays, opt_arrays,
                    seed):

    random_seed.set_random_seed(seed)

    agent = agent_func()
    for i, link in enumerate(agent.links):
        set_shared_params(link, link_arrays[i])

    for i, opt in enumerate(agent.optimizers):
        set_shared_states(opt, opt_arrays[i])

    env = env_func()

    run_func(agent, env)


def run_async(n_process, agent_func, env_func, run_func):
    """Run experiments asynchronously.

    Args:
      n_process (int): number of processes
      agent_func: function that returns a new agent
      env_func: function that returns a new environment
      run_func: function that receives an agent and an environment
    """
    base_agent = agent_func()
    links = base_agent.links
    opts = base_agent.optimizers
    link_arrays = [extract_params_as_shared_arrays(link) for link in links]
    opt_arrays = [extract_states_as_shared_arrays(opt) for opt in opts]

    processes = []

    for _ in xrange(n_process):
        processes.append(mp.Process(target=run_a3c_process, args=(
            agent_func, env_func, run_func, link_arrays, opt_arrays,
            random.randint(0, 2 ** 32 - 1)
        )))

    for p in processes:
        p.start()

    for p in processes:
        p.join()
