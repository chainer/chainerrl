import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import cv2
import chainer
import os

lake = [
    [0, 0, 0, 0],
    [0, -1, 0, -1],
    [0, 0, 0, -1],
    [-1, 0, 0, 1]
]

size = 256
tris = [
    np.array([[0,0],[size//2, size//2],[0, size]], np.int32),
    np.array([[size, 0],[size//2,size//2],[size,size]], np.int32),
    np.array([[0, 0],[size//2,size//2],[size,0]], np.int32),
    np.array([[0, size],[size//2,size//2],[size,size]], np.int32)
]

tri_locs = [
    (int(size*0.05), int(size*0.55)),
    (int(size*0.6), int(size*0.55)),
    (int(size*0.3), int(size*0.2)),
    (int(size*0.3), int(size*0.9))
]

class GridEnv(gym.Env):
    def __init__(self, outdir, N=10, save_img=False):
        N=4
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(N*N)
        self.states = np.array(lake)
        #self.states = np.zeros((N, N))
        #self.states[-1, -1] = 1.0
        #self.states[-1, 2] = -1
        self.steps = 0
        self.max_steps = 30#100
        self.counts = np.zeros_like(self.states)
        self.N = N
        self.outdir = outdir
        self.save_img = save_img

        self.seed()

        try:
            os.mkdir(self.outdir + "/frames")
            os.mkdir(self.outdir + "/plots")
        except:
            pass

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

        done = reward != 0 or self.steps > self.max_steps
        self.counts[self.pos[0], self.pos[1]] += 1

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

        if reward == 0:
            reward = -0.01

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
        if not self.save_img:
            return
            
        cv2.namedWindow("values", cv2.WINDOW_NORMAL)

    def plot_sample(self, obs_space, agent, noise, ext, show=False):
        if not self.save_img:
            return

        import numpy as np
        v = []

        for i in range(obs_space):
            s = np.zeros(obs_space)
            s[i] = 1
            vals = agent.model(
                agent.batch_states([s], agent.xp, agent.phi), **{'noise': noise, 'act': True}).q_values
            vals = vals.data[0]
            vals = chainer.cuda.to_cpu(vals)
            v.append(np.asarray(vals))

        v = np.array(v).flatten()
        #v -= v.mean()
        #v /= max(abs(v.min()), v.max())
        norm_v = v.copy()
        #norm_v -= norm_v.min()
        #norm_v /= norm_v.max()
        #norm_v = norm_v * 2.0 - 1.0

        def get_color(v):
            if float(v) >= 0:
                return (0, float(min(v, 1)), 0)
            else:
                return (0, 0, float(min(abs(v), 1)))

        canvas_rows = []

        for x in range(self.N):
            row_cells = []
            for y in range(self.N):
                i = x*self.N+y
                cell = np.zeros((size, size, 3), dtype=np.float32)

                def draw_tri(pts, loc, idx, act):
                    pts = pts.reshape((-1,1,2))

                    #cv2.fillPoly(cell, [pts], get_color(norm_v[i*2+0]))
                    cv2.fillPoly(cell, [pts], get_color(norm_v[idx]))
                    cv2.putText(cell, '%.2f' % v[idx], loc,
                        cv2.FONT_HERSHEY_SIMPLEX, size/128*0.5, (1, 1, 1), size//128)

                    if act:
                        cv2.polylines(cell, [pts], True, (1, 1, 1), 5)
                    else:
                        cv2.polylines(cell, [pts], True, (0, 0, 0), 2)

                move = np.argmax(v[i*4:i*4+4])

                draw_tri(tris[0], tri_locs[0], i*4+0, move == 0)
                draw_tri(tris[1], tri_locs[1], i*4+1, move == 1)
                draw_tri(tris[2], tri_locs[2], i*4+2, move == 2)
                draw_tri(tris[3], tri_locs[3], i*4+3, move == 3)
                draw_tri(tris[move], tri_locs[move], i*4+move, True)

                """
                move = np.argmax(v[i*4:i*4+4])
                if move == 0:
                    pts = np.array([[0, 0],[size*0.1],[size,size]], np.int32)
                elif move == 1:
                    sym = "v"
                elif move == 2:
                    sym = "<"
                elif move == 3:
                    sym = ">"
                cv2.putText(cell, sym, (size//2, size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, size/128*0.5, (1, 1, 1), size//128)
                """

                cv2.putText(cell, str(self.counts[x, y]), (int(size*0.4), int(size*0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, size/128*0.5, (1, 1, 1), size//128)

                #cv2.rectangle(cell, (0, 0), (size, size), (1, 1, 1), 2)

                if self.states[x, y] > 0:
                    cv2.rectangle(cell, (0, 0), (size, size), (0, 1, 0), size//20)
                elif self.states[x, y] < 0:
                    cv2.rectangle(cell, (0, 0), (size, size), (0, 0, 1), size//20)

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

        img = np.vstack([canvas, info])

        cv2.imwrite(self.outdir + "/frames/" + "%06d" % agent.t + ext, img*255.0)

        if show:
            cv2.imshow("values", img)
            cv2.waitKey(1)

    def plot_values(self, obs_space, agent):
        self.plot_sample(obs_space, agent, False, ".png", show=True)
        #if agent.t % 1000 == 0:
        #    self.plot_sample(obs_space, agent, True, "-sample1.png")
        #    self.plot_sample(obs_space, agent, True, "-sample2.png")
        #    self.plot_sample(obs_space, agent, True, "-sample3.png")
