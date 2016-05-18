import os
import sys


class DoomEnv(object):

    def __init__(self, vizdoom_dir=os.path.expanduser('~/ViZDoom'),
                 window_visible=True, scenario='basic'):

        sys.path.append(os.path.join(vizdoom_dir, "examples/python"))
        from vizdoom import DoomGame
        from vizdoom import Mode
        from vizdoom import Button
        from vizdoom import GameVariable
        from vizdoom import ScreenFormat
        from vizdoom import ScreenResolution

        # Create DoomGame instance. It will run the game and communicate with
        # you.
        game = DoomGame()
        game.load_config(os.path.join(
            vizdoom_dir, "examples", 'config', scenario + '.cfg'))
        # Sets path to vizdoom engine executive which will be spawned as a
        # separate process. Default is "./vizdoom".
        game.set_vizdoom_path(os.path.join(vizdoom_dir, "bin/vizdoom"))
        # Sets path to doom2 iwad resource file which contains the actual doom
        # game. Default is "./doom2.wad".
        game.set_doom_game_path(
            os.path.join(vizdoom_dir, 'scenarios/freedoom2.wad'))
        # Sets path to additional resources iwad file which is basically your
        # scenario iwad. If not specified default doom2 maps will be used and
        # it's pretty much useles... unless you want to play doom.
        game.set_doom_scenario_path(
            os.path.join(vizdoom_dir, 'scenarios', scenario + '.wad'))
        # Sets map to start (scenario .wad files can contain many maps).
        game.set_doom_map("map01")
        # Sets resolution. Default is 320X240
        # game.set_screen_resolution(ScreenResolution.RES_160X120)
        # # game.set_screen_resolution(ScreenResolution.RES_640X480)
        # # Sets the screen buffer format. Not used here but now you can change
        # # it. Defalut is CRCGCB.
        game.set_screen_format(ScreenFormat.RGB24)
        #
        # # Sets other rendering options
        # game.set_render_hud(False)
        # game.set_render_crosshair(False)
        # game.set_render_weapon(True)
        # game.set_render_decals(False)
        # game.set_render_particles(False)
        #
        # # Adds buttons that will be allowed.
        # if scenario == 'basic.wad':
        #     game.add_available_button(Button.MOVE_LEFT)
        #     game.add_available_button(Button.MOVE_RIGHT)
        #     game.add_available_button(Button.ATTACK)
        #     # Causes episodes to finish after 200 tics (actions)
        #     game.set_episode_timeout(200)
        #     # Sets the livin reward (for each move) to -1
        #     game.set_living_reward(-1)
        #
        # elif scenario == 'health_gathering.wad':
        #     game.add_available_button(Button.MOVE_LEFT)
        #     game.add_available_button(Button.MOVE_RIGHT)
        #     game.add_available_button(Button.MOVE_FORWARD)
        #
        # # Adds game variables that will be included in state.
        # game.add_available_game_variable(GameVariable.AMMO2)

        # Makes the window appear (turned on by default)
        game.set_window_visible(window_visible)

        # # Turns on the sound. (turned off by default)
        # game.set_sound_enabled(True)

        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR,
        # PLAYER mode is default)
        # game.set_mode(Mode.PLAYER)

        # Initialize the game. Further configuration won't take any effect from
        # now on.
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
        r = self.game.make_action(self.actions[action], 5)
        r /= 100
        return self.game.get_state(), r, self.game.is_episode_finished(), None
