import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import cv2
import chainer

class GridEnv(gym.Env):
    def __init__(self, N=10):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(N*N)
        self.states = np.zeros((N, N))
        self.states[-1, -1] = 1.0
        self.states[-1, 2] = -1
        self.steps = 0
        self.N = N

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def one_hot(self, pos):
        state = np.zeros((self.N, self.N))
        state[pos[0], pos[1]] = 1
        state = state.flatten()

        return state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        reward = self.states[self.pos[0], self.pos[1]]
        self.steps += 1

        done = reward != 0 or self.steps > 100

        if action == 0:
            self.pos[1] -= 1
        elif action == 1:
            self.pos[1] += 1
        if action == 2:
            self.pos[0] -= 1
        elif action == 3:
            self.pos[0] += 1

        self.pos = np.clip(self.pos, 0, self.N-1)
        state = self.one_hot(self.pos)

        return state, reward, done, {}

    def reset(self):
        self.pos = np.array([0, 0])
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
        norm_v = norm_v * 2.0 - 1.0

        def get_color(v):
            if float(v) >= 0:
                return (0, float(v), 0)
            else:
                return (0, 0, float(abs(v)))

        canvas_rows = []
        size = 64

        for x in range(self.N):
            row_cells = []
            for y in range(self.N):
                i = x*self.N+y
                cell = np.zeros((size, size, 3), dtype=np.float32)

                def draw_tri(pts, loc, idx):
                    pts = pts.reshape((-1,1,2))
                    cv2.fillPoly(cell, [pts], get_color(norm_v[idx]))
                    #cv2.fillPoly(cell, [pts], get_color(norm_v[i*2+0]))
                    cv2.putText(cell, '%.2f' % v[idx], loc,
                        cv2.FONT_HERSHEY_SIMPLEX, size/128*0.5, (1, 1, 1), size//128)

                pts = np.array([[0,0],[size//2, size//2],[0, size]], np.int32)
                loc = (int(size*0.1), int(size*0.55))
                draw_tri(pts, loc, i*4+0)

                val = v[i*4+1]
                pts = np.array([[size, 0],[size//2,size//2],[size,size]], np.int32)
                loc = (int(size*0.6), int(size*0.55))
                draw_tri(pts, loc, i*4+1)

                val = v[i*4+2]
                pts = np.array([[0, 0],[size//2,size//2],[size,0]], np.int32)
                loc = (int(size*0.3), int(size*0.2))
                draw_tri(pts, loc, i*4+2)

                val = v[i*4+3]
                pts = np.array([[0, size],[size//2,size//2],[size,size]], np.int32)
                loc = (int(size*0.3), int(size*0.8))
                draw_tri(pts, loc, i*4+3)

                move = np.argmax(v[i*4:i*4+3])
                if move == 0:
                    pt = []

                cv2.rectangle(cell, (0, 0), (size, size), (1, 1, 1), 2)

                row_cells.append(cell)
            canvas_rows.append(np.hstack(row_cells))

        canvas = np.vstack(canvas_rows)
        width = canvas.shape[1]
        info = np.zeros((size, width, 3))

        text = ''
        text += 't: %s   ' % agent.t

        try:
            text += 'epsilon: %.2f' % agent.explorer.epsilon
        except:
            pass

        cv2.putText(info, text, (100, size//2),
            cv2.FONT_HERSHEY_SIMPLEX, size/128*0.5, (1, 1, 1), size//128)

        cv2.imshow("values", np.vstack([canvas, info]))
        cv2.waitKey(1)
