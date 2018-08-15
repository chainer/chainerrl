import sys
sys.path.insert(0, ".")

from chainerrl import explorers
from chainerrl import experiments
from chainerrl import agent
from chainerrl.replay_buffer import ReplayUpdater
from chainerrl import replay_buffer
from chainerrl.replay_buffer import batch_experiences
from chainerrl.misc.batch_states import batch_states
from chainerrl import misc
import argparse

import numpy as np
from grid_env import GridEnv


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='results',
                    help='Directory path to save output files.'
                         ' If it does not exist, it will be created.')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed [0, 2 ** 31)')
parser.add_argument('--thompson', action='store_true', default=False)
parser.add_argument('--entropy-coef', type=float, default=0)
args = parser.parse_args()

ent = args.entropy_coef
outdir = args.outdir

misc.set_random_seed(22)

class Table(agent.AttributeSavingMixin, agent.Agent):
    saved_attributes = ()

    def __init__(self, vis):
        self.last_state = None
        self.last_action = None
        self.replay_buffer = replay_buffer.ReplayBuffer(1000)
        self.t = 0
        self.q_mu = np.random.normal(size=(16, 4))
        self.q_sigma = np.ones((16, 4))

        #self.q_mu_target = np.random.normal(size=(16, 4))

        self.xp = np
        self.phi = lambda x: x
        self.vis = vis

        self.replay_updater = ReplayUpdater(
            replay_buffer=self.replay_buffer,
            update_func=self.update,
            batchsize=32,
            episodic_update=False,
            episodic_update_len=-1,
            n_times_update=1,
            replay_start_size=100,
            update_interval=1,
        )

        vis.init_plot()

    def act(self, obs):
        action = self.q_mu[obs.argmax()].argmax()
        return action

    def update(self, experiences, errors_out=None):
        exp_batch = batch_experiences(experiences, xp=self.xp, phi=self.phi,
                                      batch_states=batch_states)
        s = exp_batch['state'].argmax(axis=1)
        s_n = exp_batch['next_state'].argmax(axis=1)
        a = exp_batch['action']
        term = exp_batch['is_state_terminal']

        lr = 0.1
        gamma = 0.9

        mu_target = exp_batch['reward'] + gamma * (1-term) * self.q_mu[s_n].max(axis=1)
        sigma_target = gamma * self.q_sigma[s_n, self.q_mu[s_n].argmax(axis=1)]
        sigma_entropy_grad = ent * 2.0 / np.sum(self.q_sigma)
        self.q_sigma[s, a] += lr * (sigma_target - self.q_sigma[s, a])
        self.q_sigma += sigma_entropy_grad
        self.q_mu[s, a] += lr * (mu_target - self.q_mu[s, a])

    def act_and_train(self, obs, reward):
        if args.thompson:
            noise = self.q_sigma[obs.argmax()]
            vals = self.q_mu[obs.argmax()] + np.random.normal(size=noise.shape) * noise
            action = vals.argmax()
        else:
            action = self.q_mu[obs.argmax()].argmax()
            if np.random.uniform() < 0.1:
                action = np.random.randint(4)

        self.t += 1

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        if self.t % 100 == 0:
            self.vis.plot_values(len(obs), self)

        if self.t % 100 == 0:
            self.plot_samples()

        #if self.t % 1000 == 0:
        #    self.q_mu_target = self.q_mu.copy()

        return self.last_action

    def plot_samples(self):
        import matplotlib.pyplot as plt
        import numpy as np
        g = 0.9
        trueq = [
            g**6, g**5, g**6, g**6,
            -g, -1, -g, -1,
            g**4, g**3, -g, -1,
            -1, g**2, g, 1]

        plt.title("Q(S, right) @ step: %s" % self.t)
        plt.xlim((-1, 1))

        plt.plot(trueq, -np.arange(16), "^", label="true Q")

        spread = self.q_sigma[:, 1]
        center = self.q_mu[:, 1]

        plt.errorbar(center, -np.arange(16), xerr=spread, fmt='ok', lw=2)

        plt.legend()
        #plt.show()
        plt.savefig(outdir + "/plots/" + "%06d" % self.t + ".png")
        plt.clf()

    def stop_episode_and_train(self, state, reward, done=False):
        assert self.last_state is not None
        assert self.last_action is not None

        # Add a transition to the replay buffer
        self.replay_buffer.append(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=state,
            next_action=self.last_action,
            is_state_terminal=done)

        self.stop_episode()

    def stop_episode(self):
        self.last_state = None
        self.last_action = None

        self.replay_buffer.stop_current_episode()

    def get_statistics(self):
        return [('oof', 100)]

env = GridEnv(outdir, -1, save_img=True)
eval_env = GridEnv(outdir, -1, save_img=False)
agent = Table(env)

eval_explorer = explorers.Greedy()
experiments.train_agent_with_evaluation(
    agent=agent, env=env, steps=10000,
    eval_n_runs=1, eval_interval=10,
    outdir=outdir, eval_explorer=eval_explorer,
    max_episode_len=1000,
    eval_env=eval_env,
    save_best_so_far_agent=False,
)
