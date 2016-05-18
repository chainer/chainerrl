import os
import sys


class DoomEnv(object):

    def __init__(self, vizdoom_dir=os.path.expanduser('~/ViZDoom'),
                 window_visible=True, scenario='basic', skipcount=10):

        self.skipcount = skipcount

        sys.path.append(os.path.join(vizdoom_dir, "examples/python"))
        from vizdoom import DoomGame
        from vizdoom import ScreenFormat
        from vizdoom import ScreenResolution

        game = DoomGame()

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
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_window_visible(window_visible)

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
        return self.game.get_state(), r, self.game.is_episode_finished(), None
