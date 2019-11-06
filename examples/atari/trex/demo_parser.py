from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import collections
import os

import cv2
cv2.ocl.setUseOpenCL(False)  # NOQA
import gym
import numpy as np

from chainerrl.wrappers import atari_wrappers
from chainerrl.wrappers.score_mask_atari import AtariMask

def str_to_bool(value):
    if value == 'True':
         return True
    elif value == 'False':
         return False
    else:
         raise ValueError("{} is not a bool!".format(value))

class AtariGrandChallengeParser():
    """Parses Atari Grand Challenge data.

    See https://arxiv.org/abs/1705.10998.
    """

    def __init__(self, src, env):
        tmp_env = env
        self.game = env.game.replace("_", "")
        if self.game == "montezumarevenge":
            self.game = "revenge"

        self.screens_dir = os.path.join(src, "screens", self.game)
        self.trajectories_dir = os.path.join(src, "trajectories", self.game)
        self.mask = AtariMask(env)
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

        self.trajectories, self.screens = self.preprocess(trajectories, screens)

        episodes = []
        for episode, screens in zip(self.trajectories, self.screens):
            current_episode = []
            for i in range(len(screens) - 1):
                obs = screens[i]
                a = episode['action'][i]
                r = episode['reward'][i+1]
                new_obs = screens[i + 1]
                done = episode['terminal'][i+1]
                info = {}
                current_episode.append(
                    {"obs" : obs,
                    "action" : a,
                    "reward" : r,
                    "new_obs" : new_obs,
                    "done" : done,
                    "info" : info})
                if done:
                    break
            episodes.append(current_episode)
            current_episode = []
        self.episodes = episodes

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
                     'terminal': str_to_bool,
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

        demos = non_duplicates[start:num_demos*skip + start:skip]
        assert len(demos) == num_demos
        print("(traj_num, score) pairs: ", demos)
        return demos

    def preprocess(self, trajectories, screens):
        assert [len(traj['frame']) for traj in trajectories] == [len(ep_screens) for ep_screens in screens]
        # apply score mask (score_mask_atari.py)
        new_screens = []
        new_trajs = []
        for i in range(len(screens)):
            episode_screens = screens[i]
            trajectory = trajectories[i]
            new_ep_screens = []
            new_traj = {'frame' : [], 'reward' : [], 'score' : [], 'terminal' : [], 'action' : []}
            for k in range(len(episode_screens)):
                new_ep_screens.append(self.mask(episode_screens[k]))

            # Max and skip
            obs_buffer = np.zeros((2,) + new_ep_screens[0].shape, dtype=np.uint8)
            tmp_new_screens = []
            total_reward = 0
            for j in range(len(new_ep_screens)):
                total_reward += trajectory['reward'][j]
                if j % 4 == 0:
                    obs_buffer[0] = new_ep_screens[j]
                if j % 4 == 1:
                    obs_buffer[1] = new_ep_screens[j]
                if j % 4 == 3 or trajectory['terminal'][j]:
                    max_frame = obs_buffer.max(axis=0)
                    tmp_new_screens.append(max_frame)
                    new_traj['frame'].append(int(i / 4))
                    new_traj['reward'].append(total_reward)
                    if new_traj['score']:
                        new_traj['score'].append(new_traj['score'][-1] + total_reward)
                    else:
                        new_traj['score'] = [total_reward]
                    total_reward = 0
                    # Note that the observation on the done=True frame
                    # doesn't matter
                    new_traj['terminal'].append(trajectory['terminal'][j])
                    new_traj['action'].append(trajectory['action'][j-3])
                    if trajectory['terminal'][j]:
                        break
            assert len(tmp_new_screens) == len(new_traj['frame'])
            assert int(len(new_ep_screens)/4) <= len(tmp_new_screens) <= int(len(new_ep_screens)/4) + 1
            assert new_traj['score'][-1] == trajectory['score'][-1]
            new_ep_screens = tmp_new_screens

            # grayscale, resize
            for l in range(len(new_ep_screens)):
                new_ep_screens[l] = cv2.cvtColor(new_ep_screens[l], cv2.COLOR_RGB2GRAY)
                new_ep_screens[l] = cv2.resize(new_ep_screens[l], (84, 84),
                           interpolation=cv2.INTER_AREA)

            # Framestack
            stacked_frames = collections.deque([], maxlen=4)
            stack_axis = {'hwc': 2, 'chw': 0}['hwc']
            tmp_new_screens = []
            for _ in range(4):
                stacked_frames.append(np.expand_dims(new_ep_screens[0], 0))
            for m in range(len(new_ep_screens)):
                tmp_new_screens.append(atari_wrappers.LazyFrames(list(stacked_frames),
                                       stack_axis=0))
                stacked_frames.append(np.expand_dims(new_ep_screens[m], 0))
            new_ep_screens = tmp_new_screens
            
            new_screens.append(new_ep_screens)
            new_trajs.append(new_traj)
        return new_trajs, new_screens

def main():
    env = "SpaceInvadersNoFrameskip-v4"
    atari_env = gym.make(env)
    src = "/Users/prabhat/Downloads/atari_v1/"
    thing = AtariGrandChallengeParser(src, atari_env)
    thing.parse_trajectory(351)


if __name__ == '__main__':
    main()
