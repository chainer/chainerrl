from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import os

import cv2
cv2.ocl.setUseOpenCL(False)  # NOQA
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
                    if self.game == "montezumarevenge":
                        self.game = "revenge"
                    break
                else:
                    tmp_env = env.env
            except Exception as error:
                print("Error: " + error)
                print("Env is not an Atari env.")
        self.screens_dir = os.path.join(src, "screens", self.game)
        self.trajectories_dir = os.path.join(src, "trajectories", self.game)
        traj_numbers, traj_scores = zip(*self.get_sorted_traj_indices())
        assert isinstance(traj_numbers, tuple)
        assert isinstance(traj_scores, tuple)
        traj_numbers = list(traj_numbers)
        traj_scores = list(traj_scores)
        trajectories = [self.parse_trajectory(traj_num) for traj_num in traj_numbers]
        assert traj_scores == [traj['score'][-1] for traj in trajectories]
        assert traj_scores == [sum(traj['reward']) for traj in trajectories]
        screens = [self.parse_screens(traj_num) for traj_num in traj_numbers]
        assert len(screens) == len(trajectories)
        # parse screens, apply preprocessing, port to demonstration format
        # Apply masks

    def parse_screens(self, traj_number):
        # add screens
        traj_screens_dir= os.path.join(self.screens_dir, str(traj_number))
        screens = []
        screen_files = [f for f in os.listdir(traj_screens_dir) \
                        if os.path.isfile(os.path.join(traj_screens_dir, f))]
        screen_files.sort(key=lambda x : int(os.path.splitext(os.path.basename(x))[0]))
        assert [int(os.path.splitext(os.path.basename(x))[0]) for x in screen_files] == np.arange(len(screen_files)).tolist()
        for frame_file in screen_files:
            screens.append(cv2.imread(os.path.join(traj_screens_dir, frame_file)))

        return screens

    def parse_trajectory(self, traj_number):
        # check if that trajectory number exists, then parse
        trajectory_file = os.path.join(self.trajectories_dir,
                                       str(traj_number) + ".txt")
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
        return episode

    def get_sorted_traj_indices(self):
        # This function is adapted from https://github.com/hiwonjoon/ICML2019-TREX

        # need to pick out a subset of demonstrations based on desired performance
        # first let's sort the demos by performance, we can use the trajectory number to index into the demos so just
        # need to sort indices based on 'score'
        # Note, we're only keeping the full demonstrations that end in terminal to avoid people who quit before the game was over
        traj_nums = []
        traj_scores = []
        # 
        files = [f for f in os.listdir(self.trajectories_dir) if os.path.isfile(os.path.join(self.trajectories_dir, f))]
        trajectories = [int(os.path.splitext(os.path.basename(file))[0]) for file in files]
        for traj_number in trajectories:
            episode = self.parse_trajectory(traj_number)
            if self.game == "revenge":
                traj_nums.append(traj_number)
                traj_scores.append(episode['score'][-1])
            elif episode['terminal'][-1]:
                traj_nums.append(traj_number)
                traj_scores.append(episode['score'][-1])

        sorted_traj_nums = [x for _, x in sorted(zip(traj_scores, traj_nums), key=lambda pair: pair[0])]
        sorted_traj_scores = sorted(traj_scores)

        print("Sorted trajectory scores " + str(sorted_traj_scores))
        print("Max human score", max(sorted_traj_scores))
        print("Min human score", min(sorted_traj_scores))

        seen_scores = set()
        non_duplicates = []
        for i, s in zip(sorted_traj_nums, sorted_traj_scores):
            if s not in seen_scores:
                seen_scores.add(s)
                non_duplicates.append((i,s))
        print("Number of unduplicated scores", len(seen_scores))
        num_demos = 12
        if self.game == "spaceinvaders":
            start = 0
            skip = 4
        elif self.game == "revenge":
            start = 0
            skip = 1
        elif self.game == "qbert":
            start = 0
            skip = 3
        elif self.game == "mspacman":
            start = 0
            skip = 3
        elif self.game == "pinball":
            start = 0
            skip = 1
        elif self.game == "revenge":
            start = 0
            skip = 1

        demos = non_duplicates[start:num_demos*skip + start:skip]
        assert len(demos) == num_demos
        print("(traj_num, score) pairs: ", demos)
        return demos

def main():
    env = "SpaceInvadersNoFrameskip-v4"
    atari_env = gym.make(env)
    src = "/Users/prabhat/Downloads/atari_v1/"
    thing = AtariGrandChallengeParser(src, atari_env)
    thing.parse_trajectory(351)


if __name__ == '__main__':
    main()
