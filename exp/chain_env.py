import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import cv2
import chainer

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

    def init_plot(self):
        cv2.namedWindow("values", cv2.WINDOW_NORMAL)

    def plot_values(self, obs_space, agent):
        import numpy as np
        v = []

        for i in range(obs_space):
            s = np.zeros(obs_space)
            s[i] = 1
            vals = agent.model(
                agent.batch_states([s], agent.xp, agent.phi), **{'noise': False}).q_values
            vals = vals.data[0]
            vals = chainer.cuda.to_cpu(vals)
            v.append(np.asarray(vals))

        v = np.array(v).flatten()
        #v -= v.mean()
        #v /= max(abs(v.min()), v.max())
        norm_v = v.copy()
        norm_v -= norm_v.min()
        norm_v /= norm_v.max()
        #v = v * 2.0 - 1.0

        def get_color(v):
            if float(v) >= 0:
                return (0, float(v), 0)
            else:
                return (0, 0, float(abs(v)))

        canvas = []
        size = 256

        for i in range(obs_space):
            cell = np.zeros((size, size, 3), dtype=np.float32)

            val = v[i*2+0]
            pts = np.array([[0,0],[size//2, size//2],[0, size]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(cell, [pts], get_color(norm_v[i*2+0]))
            cv2.putText(cell, '%.2f' % val, (int(size*0.1), int(size*0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, size/128*0.5, (1, 1, 1), size//128)

            val = v[i*2+1]
            pts = np.array([[size, 0],[size//2,size//2],[size,size]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(cell, [pts], get_color(norm_v[i*2+1]))
            cv2.putText(cell, '%.2f' % val, (int(size*0.6), int(size*0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, size/128*0.5, (1, 1, 1), size//128)

            cv2.rectangle(cell, (0, 0), (size-5, size-5), (1, 1, 1), 5)

            canvas.append(cell)

        width = len(canvas)*size
        info = np.zeros((size, width, 3))

        text = ''
        text += 't: %s   ' % agent.t

        try:
            text += 'epsilon: %.2f' % agent.explorer.epsilon
        except:
            pass

        cv2.putText(info, text, (100, size//2),
            cv2.FONT_HERSHEY_SIMPLEX, size/128*0.5, (1, 1, 1), size//128)

        cv2.imshow("values", np.vstack([np.hstack(canvas), info]))
        cv2.waitKey(1)
