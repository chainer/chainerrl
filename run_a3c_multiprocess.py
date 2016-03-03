import multiprocessing as mp
import os
import argparse
import random

import numpy as np

import chainer
from chainer import optimizers

import policy
import v_function
import a3c
import xor


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


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
                param.get_obj(), dtype=old_param.dtype).reshape(old_param.shape)


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


def run_a3c_process(pi_shared_arrays, v_shared_arrays, opt_shared_arrays):

    set_random_seed(os.getpid())

    pi = policy.FCSoftmaxPolicy(2, 2, 10, 2)
    v = v_function.FCVFunction(2, 10, 2)

    set_shared_params(pi, pi_shared_arrays)
    set_shared_params(v, v_shared_arrays)

    optimizer = optimizers.RMSprop(lr=1e-3)
    optimizer.setup(chainer.ChainList(pi, v))
    set_shared_states(optimizer, opt_shared_arrays)
    agent = a3c.A3C(pi, v, optimizer, 1, 0.9)
    env = xor.XOR()

    total_r = 0

    for i in xrange(2000):
        is_terminal = i % 2 == 1
        action = agent.act(env.state, env.reward, is_terminal)
        env.receive_action(action)
        total_r += env.reward

    print 'pid:{}, total_r:{}'.format(os.getpid(), total_r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--share-pi', action='store_true', default=False)
    parser.add_argument('--share-v', action='store_true', default=False)
    parser.add_argument('--share-opt', action='store_true', default=False)
    args = parser.parse_args()

    shared_pi = policy.FCSoftmaxPolicy(2, 2, 10, 2)
    shared_v = v_function.FCVFunction(2, 10, 2)
    pi_shared_arrays = extract_params_as_shared_arrays(
        shared_pi) if args.share_pi else {}
    v_shared_arrays = extract_params_as_shared_arrays(
        shared_v) if args.share_v else {}
    shared_optimizer = optimizers.RMSprop()
    shared_optimizer.setup(chainer.ChainList(shared_pi, shared_v))
    opt_shared_arrays = extract_states_as_shared_arrays(
        shared_optimizer) if args.share_opt else {}

    threads = []

    for _ in xrange(args.processes):

        threads.append(mp.Process(target=run_a3c_process, args=(
            pi_shared_arrays, v_shared_arrays, opt_shared_arrays)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
