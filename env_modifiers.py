from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
def make_rendered(env, *render_args, **render_kwargs):
    base_step = env.step
    base_close = env.close

    def step(action):
        ret = base_step(action)
        env.render(*render_args, **render_kwargs)
        return ret

    def close():
        env.render(*render_args, close=True, **render_kwargs)
        base_close()

    env.step = step
    env.close = close


def make_timestep_limited(env, timestep_limit):
    t = [1]
    old_step = env.step
    old_reset = env.reset

    def step(action):
        observation, reward, done, info = old_step(action)
        if t[0] >= timestep_limit:
            done = True
        t[0] += 1
        return observation, reward, done, info

    def reset():
        t[0] = 1
        return old_reset()

    env.step = step
    env.reset = reset


def make_action_filtered(env, action_filter):
    old_step = env.step

    def step(action):
        return old_step(action_filter(action))

    env.step = step


def make_reward_filtered(env, reward_filter):
    old_step = env.step

    def step(action):
        observation, reward, done, info = old_step(action)
        reward = reward_filter(reward)
        return observation, reward, done, info

    env.step = step
