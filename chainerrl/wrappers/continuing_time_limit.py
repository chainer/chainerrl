from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import logging
import time

import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder


class ContinuingTimeLimit(gym.Wrapper):
    """TimeLimit wrapper for continuing environments.

    This is similar gym.wrappers.TimeLimit, which sets a time limit for
    each episode, except that done=False is returned and that
    info['needs_reset'] is set to True when past the limit.

    Code that calls env.step is responsible for checking the info dict, the
    fourth returned value, and resetting the env if it has the 'needs_reset'
    key and its value is True.

    Args:
        env (gym.Env): Env to wrap.
        max_episode_steps (int): Maximum number of timesteps during an episode,
            after which the env needs a reset.
    """

    def __init__(self, env, max_episode_steps):
        super(ContinuingTimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None,\
            "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._max_episode_steps <= self._elapsed_steps:
            info['needs_reset'] = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


class ContinuingTimeLimitMonitor(Monitor):
    """`Monitor` with ChainerRL's `ContinuingTimeLimit` support.

    Because of the original implementation's design,
    explicit `close()` is needed to save the last episode.
    Do not forget to call `close()` at the last line of your script.

    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
    """

    def _start(self, directory, video_callable=None, force=False, resume=False,
               write_upon_reset=False, uid=None, mode=None):
        if self.env_semantics_autoreset:
            raise gym.error.Error(
                "Detect 'semantics.autoreset=True' in `env.metadata`, "
                "which means the env comes from deprecated OpenAI Universe.")
        ret = super()._start(directory=directory,
                             video_callable=video_callable, force=force,
                             resume=resume, write_upon_reset=write_upon_reset,
                             uid=uid, mode=mode)
        if self.env.spec is None:
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id
        self.stats_recorder = _ContinuingTimeLimitStatsRecorder(
            directory,
            '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix),
            autoreset=False, env_id=env_id)
        return ret


class _ContinuingTimeLimitStatsRecorder(StatsRecorder):
    """`StatsRecorder` with ChainerRL's `ContinuingTimeLimit` support.

    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/stats_recorder.py
    """

    def __init__(self, directory, file_prefix, autoreset=False, env_id=None):
        super().__init__(directory, file_prefix,
                         autoreset=autoreset, env_id=env_id)
        self._save_completed = True

    def before_reset(self):
        assert not self.closed

        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)

        if self.done is not None and not self.done and self.steps > 0:
            self.logger.debug('Tried to reset env which is not done. '
                              'StatsRecorder completes the last episode.')
            self.save_complete()

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_step(self, observation, reward, done, info):
        self._save_completed = False
        return super().after_step(observation, reward, done, info)

    def save_complete(self):
        if not self._save_completed:
            super().save_complete()
            self._save_completed = True

    def close(self):
        self.save_complete()
        super().close()
