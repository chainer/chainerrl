from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from multiprocessing import Pipe
from multiprocessing import Process

from cached_property import cached_property
import numpy as np

from chainerrl import env


def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'spec':
            remote.send(env.spec)
        elif cmd == 'seed':
            remote.send(env.seed(data))
        else:
            raise NotImplementedError


class VectorEnv(env.Env):

    def __init__(self, env_fns):
        """envs: list of gym environments to run in subprocesses

        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = \
            [Process(target=worker, args=(work_remote, env_fn))
             for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.last_obs = [None] * self.num_envs
        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

    def __del__(self):
        self.close()

    @cached_property
    def spec(self):
        self.remotes[0].send(('spec', None))
        spec = self.remotes[0].recv()
        return spec

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        self.last_obs, rews, dones, infos = zip(*results)
        return self.last_obs, rews, dones, infos

    def reset(self, mask=None):
        if mask is None:
            mask = np.zeros(self.num_envs)
        for m, remote in zip(mask, self.remotes):
            if not m:
                remote.send(('reset', None))

        obs = [remote.recv() if not m else o for m, remote,
               o in zip(mask, self.remotes, self.last_obs)]
        self.last_obs = obs
        return obs

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def seed(self, seeds=None):
        if seeds is not None:
            if isinstance(seeds, int):
                seeds = [seeds] * self.num_envs
            elif isinstance(seeds, list):
                if len(seeds) != self.num_envs:
                    raise ValueError(
                        "length of seeds must be same as num_envs {}"
                        .format(self.num_envs))
            else:
                raise TypeError(
                    "Type of Seeds {} is not supported.".format(type(seeds)))
        else:
            seeds = [None] * self.num_envs

        for remote, seed in zip(self.remotes, seeds):
            remote.send(('seed', seed))
        results = [remote.recv() for remote in self.remotes]
        return results

    @property
    def num_envs(self):
        return len(self.remotes)
