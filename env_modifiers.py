def make_rendered(env, *render_args, **render_kwargs):
    base_step = env._step

    def _step(action):
        ret = base_step(action)
        env.render(*render_args, **render_kwargs)
        return ret

    env._step = _step


def make_timestep_limited(env, timestep_limit):
    t = 1
    old__step = env._step
    old__reset = env._reset

    def _step(action):
        nonlocal t
        observation, reward, done, info = old__step(action)
        if t >= timestep_limit:
            done = True
        t += 1
        return observation, reward, done, info

    def _reset():
        nonlocal t
        t = 1
        return old__reset()

    env._step = _step
    env._reset = _reset


def make_action_filtered(env, action_filter):
    old_step = env.step

    def step(action):
        return old_step(action_filter(action))

    env.step = step


def make_reward_filtered(env, reward_filter):
    old__step = env._step

    def _step(action):
        observation, reward, done, info = old__step(action)
        reward = reward_filter(reward)
        return observation, reward, done, info

    env._step = _step
