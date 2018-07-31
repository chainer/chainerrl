import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class ChainEnv(gym.Env):
    def __init__(self, N=10):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(N)
        self.pos = 2
        self.states = [0.001]
        self.states += [0] * (N-2)
        self.states += [1.0]
        self.N = N
        self.steps = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def one_hot(self, pos):
        state = np.zeros(self.N)
        state[pos] = 1

        return state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        reward = self.states[self.pos]
        self.steps += 1

        done = self.pos == 0 or self.pos == self.N-1 or self.steps > 100

        if action == 0:
            self.pos -= 1
        else:
            self.pos += 1
        self.pos = min(self.N-1, max(0, self.pos))

        state = self.one_hot(self.pos)

        return state, reward, done, {}

    def reset(self):
        self.pos = 2
        self.steps = 0
        state = self.one_hot(self.pos)

        return state

    def render(self, mode='human'):
        return None

    def close(self):
        pass
