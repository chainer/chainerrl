from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import os

import gym
import numpy as np

from pdb import set_trace

class AtariGrandChallengeParser():
    """Parses Atari Grand Challenge data.

    See https://arxiv.org/abs/1705.10998.
    """

    def __init__(self, src, env):
        # check whether atari env
        tmp_env = env
        while True:
            try:
                if isinstance(tmp_env, gym.envs.atari.atari_env.AtariEnv):
                    self.game = env.game.replace("_", "")
                    break
                else:
                    tmp_env = env.env
            except Exception as error:
                print("Error: " + error)
                print("Env is not an Atari env.")
        self.screens_dir = os.path.join(src, "screens", self.game)
        self.trajectories_dir = os.path.join(src, "trajectories", self.game)

    def parse(self, traj_number):
        # check if that trajectory number exists, then parse
        trajectory_file = os.path.join(self.trajectories_dir,
                                       str(traj_number) + ".txt")
        screens_dir = os.path.join(self.trajectories_dir, str(traj_number))

        traj_file_lines = open(trajectory_file, "r").readlines()
        entries = ''.join(str.split(traj_file_lines[1])).split(",")
        assert entries == ['frame', 'reward', 'score', 'terminal', 'action']
        type_dict = {'frame': int,
                     'reward': int,
                     'score': int,
                     'terminal': bool,
                     'action': int}
        episode = dict()
        for entry in entries:
            episode[entry] = []
        for i in range (2, len(traj_file_lines)):
            data = traj_file_lines[i]
            data_points = ''.join(str.split(data)).split(",")
            for k in range(len(data_points)):
                entry = entries[k]
                data_point = data_points[k]
                episode[entry].append(type_dict[entry](data_point))
        assert episode['frame'][0] == 0

        set_trace()

def main():
    env = "SpaceInvadersNoFrameskip-v4"
    atari_env = gym.make(env)
    src = "/home/prabhat/Downloads/atari_v1/"
    thing = AtariGrandChallengeParser(src, atari_env)
    thing.parse(351)


if __name__ == '__main__':
    main()
