from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import multiprocessing as mp
import warnings

import chainer
import numpy as np


class AbnormalExitWarning(Warning):
    """Warning category for abnormal subprocess exit."""
    pass


def ensure_initialized_update_rule(param):
    u = param.update_rule
    if u.state is None:
        u._state = {}
        """FIXME: UpdateRule.state is read-only.

        But, force u.state = {}.
        """

        u.init_state(param)


def set_shared_params(a, b):
    """Set shared params (and persistent values) to a link.

    Args:
      a (chainer.Link): link whose params are to be replaced
      b (dict): dict that consists of (param_name, multiprocessing.Array)
    """
    assert isinstance(a, chainer.Link)
    for param_name, param in a.namedparams():
        if param_name in b:
            shared_param = b[param_name]
            param.array = np.frombuffer(
                shared_param, dtype=param.dtype).reshape(param.shape)
    for persistent_name in a._persistent:
        persistent = a.__dict__[persistent_name]
        if persistent_name in b:
            shared_param = b[persistent_name]
            if isinstance(persistent, np.ndarray):
                a.__dict__[persistent_name] = np.frombuffer(
                    shared_param, dtype=persistent.dtype).reshape(
                        persistent.shape)
            else:
                assert np.isscalar(persistent)
                # We wrap scalars with np.ndarray because
                # multiprocessing.RawValue cannot be used as a scalar, while
                # np.ndarray can be.
                typecode = np.asarray(persistent).dtype.char
                a.__dict__[persistent_name] = np.frombuffer(
                    shared_param, dtype=typecode).reshape(())


def make_params_not_shared(a):
    """Make a link's params not shared.

    Args:
      a (chainer.Link): link whose params are to be made not shared
    """
    assert isinstance(a, chainer.Link)
    for param in a.params():
        param.array = param.array.copy()


def assert_params_not_shared(a, b):
    assert isinstance(a, chainer.Link)
    assert isinstance(b, chainer.Link)
    a_params = dict(a.namedparams())
    b_params = dict(b.namedparams())
    for name, a_param in a_params.items():
        b_param = b_params[name]
        assert a_param.array.ctypes.data != b_param.array.ctypes.data


def set_shared_states(a, b):
    assert isinstance(a, chainer.Optimizer)
    assert hasattr(a, 'target'), 'Optimizer.setup must be called first'
    for param_name, param in a.target.namedparams():
        ensure_initialized_update_rule(param)
        state = param.update_rule.state
        for state_name, state_val in b[param_name].items():
            s = state[state_name]
            state[state_name] = np.frombuffer(
                state_val,
                dtype=s.dtype).reshape(s.shape)


def extract_params_as_shared_arrays(link):
    assert isinstance(link, chainer.Link)
    shared_arrays = {}
    for param_name, param in link.namedparams():
        typecode = param.array.dtype.char
        shared_arrays[param_name] = mp.RawArray(typecode, param.array.ravel())
    for persistent_name in link._persistent:
        persistent = link.__dict__[persistent_name]
        if isinstance(persistent, np.ndarray):
            typecode = persistent.dtype.char
            shared_arrays[persistent_name] = mp.RawArray(
                typecode, persistent.ravel())
        else:
            assert np.isscalar(persistent)
            persistent_as_array = np.asarray([persistent])
            typecode = persistent_as_array.dtype.char
            shared_arrays[persistent_name] = mp.RawArray(
                typecode, persistent_as_array)
    return shared_arrays


def share_params_as_shared_arrays(link):
    shared_arrays = extract_params_as_shared_arrays(link)
    set_shared_params(link, shared_arrays)
    return shared_arrays


def extract_states_as_shared_arrays(optimizer):
    assert isinstance(optimizer, chainer.Optimizer)
    assert hasattr(optimizer, 'target'), 'Optimizer.setup must be called first'
    shared_arrays = {}
    for param_name, param in optimizer.target.namedparams():
        shared_arrays[param_name] = {}
        ensure_initialized_update_rule(param)
        state = param.update_rule.state
        for state_name, state_val in state.items():
            shared_arrays[param_name][
                state_name] = mp.RawArray('f', state_val.ravel())
    return shared_arrays


def share_states_as_shared_arrays(optimizer):
    shared_arrays = extract_states_as_shared_arrays(optimizer)
    set_shared_states(optimizer, shared_arrays)
    return shared_arrays


def run_async(n_process, run_func):
    """Run experiments asynchronously.

    Args:
      n_process (int): number of processes
      run_func: function that will be run in parallel
    """

    processes = []

    for process_idx in range(n_process):
        processes.append(mp.Process(target=run_func, args=(process_idx,)))

    for p in processes:
        p.start()

    for process_idx, p in enumerate(processes):
        p.join()
        if p.exitcode > 0:
            warnings.warn(
                "Process #{} (pid={}) exited with nonzero status {}".format(
                    process_idx, p.pid, p.exitcode),
                category=AbnormalExitWarning,
            )
        elif p.exitcode < 0:
            warnings.warn(
                "Process #{} (pid={}) was terminated by signal {}".format(
                    process_idx, p.pid, -p.exitcode),
                category=AbnormalExitWarning,
            )


def as_shared_objects(obj):
    if isinstance(obj, tuple):
        return tuple(as_shared_objects(x) for x in obj)
    elif isinstance(obj, chainer.Link):
        return share_params_as_shared_arrays(obj)
    elif isinstance(obj, chainer.Optimizer):
        return share_states_as_shared_arrays(obj)
    elif isinstance(obj, mp.sharedctypes.Synchronized):
        return obj
    else:
        raise ValueError('')


def synchronize_to_shared_objects(obj, shared_memory):
    if isinstance(obj, tuple):
        return tuple(synchronize_to_shared_objects(o, s)
                     for o, s in zip(obj, shared_memory))
    elif isinstance(obj, chainer.Link):
        set_shared_params(obj, shared_memory)
        return obj
    elif isinstance(obj, chainer.Optimizer):
        set_shared_states(obj, shared_memory)
        return obj
    elif isinstance(obj, mp.sharedctypes.Synchronized):
        return shared_memory
    else:
        raise ValueError('')
