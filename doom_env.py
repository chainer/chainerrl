import os
import sys
import time

import numpy as np


class DoomEnv(object):

    def __init__(self, vizdoom_dir=os.path.expanduser('~/ViZDoom'),
                 window_visible=True, scenario='basic', skipcount=10,
                 resolution_width=640, sleep=0.0, seed=None):

        self.skipcount = skipcount
        self.sleep = sleep

        sys.path.append(os.path.join(vizdoom_dir, "examples/python"))
        from vizdoom import DoomGame
        from vizdoom import ScreenFormat
        from vizdoom import ScreenResolution

        game = DoomGame()

        if seed is not None:
            assert seed >= 0 and seed < 2 ** 16, \
                "ViZDoom's random seed must be represented by unsigned int"
        else:
            # Use numpy's random state
            seed = np.random.randint(0, 2 ** 16)
        game.set_seed(seed)

        # Load a config file
        game.load_config(os.path.join(
            vizdoom_dir, "examples", 'config', scenario + '.cfg'))

        # Replace default relative paths with actual paths
        game.set_vizdoom_path(os.path.join(vizdoom_dir, "bin/vizdoom"))
        game.set_doom_game_path(
            os.path.join(vizdoom_dir, 'scenarios/freedoom2.wad'))
        game.set_doom_scenario_path(
            os.path.join(vizdoom_dir, 'scenarios', scenario + '.wad'))

        # Set screen settings
        resolutions = {640: ScreenResolution.RES_640X480,
                       320: ScreenResolution.RES_320X240,
                       160: ScreenResolution.RES_160X120}
        game.set_screen_resolution(resolutions[resolution_width])
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_window_visible(window_visible)
        game.set_sound_enabled(window_visible)

        game.init()
        self.game = game

        # Use one-hot actions
        self.n_actions = game.get_available_buttons_size()
        self.actions = []
        for i in range(self.n_actions):
            self.actions.append([i == j for j in range(self.n_actions)])

    def reset(self):
        self.game.new_episode()
        return self.game.get_state()

    def step(self, action):
        r = self.game.make_action(self.actions[action], self.skipcount)
        r /= 100
        time.sleep(self.sleep * self.skipcount)
        return self.game.get_state(), r, self.game.is_episode_finished(), None
