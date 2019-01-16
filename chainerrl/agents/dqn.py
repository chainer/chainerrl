from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_reset
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater

import cv2
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')


import sys
sys.path.insert(0, ".")
from chainerrl.agents.util import estimate

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import math

pltfig = Figure(figsize=(1,1))
pltcanvas = FigureCanvas(pltfig)
pltax = pltfig.gca()

def compute_value_loss(y, t, clip_delta=True, batch_accumulator='mean'):
    """Compute a loss for value prediction problem.

    Args:
        y (Variable or ndarray): Predicted values.
        t (Variable or ndarray): Target values.
        clip_delta (bool): Use the Huber loss function if set True.
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')
    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))
    if clip_delta:
        loss_sum = F.sum(F.huber_loss(y, t, delta=1.0))
        if batch_accumulator == 'mean':
            loss = loss_sum / y.shape[0]
        elif batch_accumulator == 'sum':
            loss = loss_sum
    else:
        loss_mean = F.mean_squared_error(y, t) / 2
        if batch_accumulator == 'mean':
            loss = loss_mean
        elif batch_accumulator == 'sum':
            loss = loss_mean * y.shape[0]
    return loss


def compute_weighted_value_loss(y, t, weights,
                                clip_delta=True, batch_accumulator='mean'):
    """Compute a loss for value prediction problem.

    Args:
        y (Variable or ndarray): Predicted values.
        t (Variable or ndarray): Target values.
        weights (ndarray): Weights for y, t.
        clip_delta (bool): Use the Huber loss function if set True.
        batch_accumulator (str): 'mean' will devide loss by batchsize
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')
    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))
    if clip_delta:
        losses = F.huber_loss(y, t, delta=1.0)
    else:
        losses = F.square(y - t) / 2
    losses = F.reshape(losses, (-1,))
    loss_sum = F.sum(losses * weights)
    if batch_accumulator == 'mean':
        loss = loss_sum / y.shape[0]
    elif batch_accumulator == 'sum':
        loss = loss_sum
    return loss


class DQN(agent.AttributeSavingMixin, agent.Agent):
    """Deep Q-Network algorithm.

    Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        average_q_decay (float): Decay rate of average Q, only used for
            recording statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            recording statistics
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """

    saved_attributes = ('model', 'target_model', 'optimizer')

    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=None, replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000, clip_delta=True,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 batch_accumulator='mean', episodic_update=False,
                 episodic_update_len=None,
                 logger=getLogger(__name__),
                 batch_states=batch_states,
                 entropy=None, entropy_coef=0,
                 vis=None,
                 noisy_y=False,
                 noisy_t=False,
                 plot=False,
                 head=False,
                 use_table=False,
                 table_lr=0.01,
                 samples=1,
                 env=None,
                 video=False,
                 table_sigma=False,
                 scale_sigma=1,
                 min_sigma=0,
                 no_nn=False,
                 outdir="results",
                 algo="DQN",
                 sigma_gamma=0.9,
                 one_sigma=False,
                 fixed_sigma=False,
                 table_noise=False,
                 gym_env=None,
                 res=20):
        self.model = q_function
        self.q_function = q_function  # For backward compatibility

        self.render = False
        if self.render:
            cv2.namedWindow("vis", cv2.WINDOW_NORMAL)
        #cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        self.env = env
        self.table_sigma = table_sigma
        self.scale_sigma = scale_sigma
        self.sigma_gamma = sigma_gamma
        self.min_sigma = min_sigma
        self.no_nn = no_nn
        self.nn = not no_nn
        self.outdir = outdir
        self.algo = algo

        self.one_sigma = one_sigma
        self.fixed_sigma = fixed_sigma
        self.table_noise = table_noise

        if self.model is not None and gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            #print(type(self.model.l1.W.data))
            self.model.to_gpu(device=gpu)
            #print(type(self.model.l1.W.data))
            #try:
            #    self.model.reset_noise()
            #except:
            #    pass
        self.use_gpu = gpu is not None and gpu >= 0

        if self.nn:
            self.xp = self.model.xp
        else:
            self.xp = np

        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.clip_delta = clip_delta
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.batch_accumulator = batch_accumulator
        assert batch_accumulator in ('mean', 'sum')
        self.logger = logger
        self.batch_states = batch_states
        print(">>>", batch_states)
        self.noisy_y = noisy_y
        self.noisy_t = noisy_t
        if episodic_update:
            update_func = self.update_from_episodes
        else:
            update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            episodic_update=episodic_update,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_model = None
        self.sync_target_network()
        # For backward compatibility
        self.target_q_function = self.target_model
        self.average_q = 0
        self.average_q_decay = average_q_decay
        self.average_loss = 0
        self.average_loss_decay = average_loss_decay

        self.entropy = entropy
        self.entropy_coef = entropy_coef

        self.last_entropy = 0

        self.vis = vis
        self.video = video
        self.head = head

        if "car" not in env:
            self.video = False

        try:
            self.vis.init_plot()
            self.plot = plot
        except:
            self.plot = False

        self.conv = False
        #cv2.namedWindow('test', cv2.WINDOW_NORMAL)

        self.res = res
        self.n_states = self.res**gym_env.observation_space.low.shape[0]
        self.n_actions = gym_env.action_space.n
        self.gym_env = gym_env
        self.noise_table = self.xp.asarray(np.random.normal(size=(self.n_states, self.n_actions)))
        self.counts = np.zeros((self.n_states, gym_env.action_space.n))
        self.counts2 = np.zeros((self.n_states, gym_env.action_space.n))
        self.visited = np.zeros((self.n_states, gym_env.action_space.n))

        self.q_table_mu = self.xp.asarray(np.ones((self.n_states, self.n_actions)) * -20)
        self.q_table_var = self.xp.asarray(np.ones((self.n_states, self.n_actions)) * self.scale_sigma)
        self.last_score = ""
        self.use_table = use_table
        self.table_lr = table_lr

        self.est_error = 0
        self.samples = samples

        self.pi = [[0, 0, 0], [0, 0, 0]]

        self.vid = None

    def save(self, dirname):
        if self.vid is not None:
            self.vid.release()

        if not self.nn:
            return

        """Save internal states."""
        super(DQN, self).save(dirname)


    def sync_target_network(self):
        if not self.nn:
            return

        """Synchronize target network with current network."""
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
            call_orig = self.target_model.__call__

            def call_test(self_, x):
                with chainer.using_config('train', False):
                    return call_orig(self_, x)

            self.target_model.__call__ = call_test
        else:
            synchronize_parameters(
                src=self.model,
                dst=self.target_model,
                method=self.target_update_method,
                tau=self.soft_update_tau)

    def update(self, experiences, errors_out=None):
        """Update the model from experiences

        This function is thread-safe.
        Args:
          experiences (list): list of dicts that contains
            state: cupy.ndarray or numpy.ndarray
            action: int [0, n_action_types)
            reward: float32
            next_state: cupy.ndarray or numpy.ndarray
            next_legal_actions: list of booleans; True means legal
        Returns:
          None
        """

        import traceback
        #traceback.print_exc()


        has_weight = 'weight' in experiences[0]
        exp_batch = batch_experiences(experiences, xp=self.xp, phi=self.phi,
                                      batch_states=self.batch_states, gamma=self.gamma)
        if has_weight:
            exp_batch['weights'] = self.xp.asarray(
                [elem[0]['weight']for elem in experiences],
                dtype=self.xp.float32)
            if errors_out is None:
                errors_out = []

        if self.nn:
            loss = self._compute_loss(
                exp_batch, self.gamma, errors_out=errors_out)
            if has_weight:
                self.replay_buffer.update_errors(errors_out)

            # Update stats
            self.average_loss *= self.average_loss_decay
            self.average_loss += (1 - self.average_loss_decay) * float(loss.data)

            self.model.cleargrads()
            loss.backward()
            #self.logger.info("%s" % self.entropy[-1].sigma.W)
            #self.logger.info("%s" % self.entropy[-1].sigma.W.grad);
            self.optimizer.update()

        if "car" in self.env:
            # table
            lr = self.table_lr# * (0.1**int(self.t/100000))
            discount = self.sigma_gamma**2.0

            s = self.discretize(exp_batch['state'])
            s_n = self.discretize(exp_batch['next_state'])
            term = exp_batch['is_state_terminal']
            a = exp_batch['action']
            na = exp_batch['next_action']

            if self.algo == "DQN":
                mu_target = exp_batch['reward'] + self.gamma * (1-term) * self.q_table_mu[s_n].max(axis=1)
                sigma_target = discount * (1-term) * self.q_table_var[s_n, self.q_table_mu[s_n].argmax(axis=1)]
            else:
                mu_target = exp_batch['reward'] + self.gamma * (1-term) * self.q_table_mu[s_n, na]
                sigma_target = discount * (1-term) * self.q_table_var[s_n, na]

            #self.q_table_sigma[s_n, na] *= gamma
            #sigma_entropy_grad = ent * 2.0 / np.sum(self.q_table_sigma)
            self.q_table_var[s, a] += lr * (sigma_target - self.q_table_var[s, a])
            #self.q_table_sigma += sigma_entropy_grad

            self.q_table_mu[s, a] += lr * (mu_target - self.q_table_mu[s, a])

    def discretize(self, states):
        inds = []
        for state in states:
            ind = self.to_state_index(state)
            inds.append(ind)

        return self.xp.asarray(inds)

    def input_initial_batch_to_target_model(self, batch):
        self.target_model(batch['state'])

    def update_from_episodes(self, episodes, errors_out=None):
        has_weights = isinstance(episodes, tuple)
        if has_weights:
            episodes, weights = episodes
            if errors_out is None:
                errors_out = []
        if errors_out is None:
            errors_out_step = None
        else:
            del errors_out[:]
            for _ in episodes:
                errors_out.append(0.0)
            errors_out_step = []

        with state_reset(self.model), state_reset(self.target_model):
            loss = 0
            tmp = list(reversed(sorted(
                enumerate(episodes), key=lambda x: len(x[1]))))
            sorted_episodes = [elem[1] for elem in tmp]
            indices = [elem[0] for elem in tmp]  # argsort
            max_epi_len = len(sorted_episodes[0])
            for i in range(max_epi_len):
                transitions = []
                weights_step = []
                for ep, index in zip(sorted_episodes, indices):
                    if len(ep) <= i:
                        break
                    transitions.append(ep[i])
                    if has_weights:
                        weights_step.append(weights[index])
                batch = batch_experiences(
                    [transitions],
                    xp=self.xp,
                    phi=self.phi,
                    gamma=self.gamma,
                    batch_states=self.batch_states)
                if i == 0:
                    self.input_initial_batch_to_target_model(batch)
                if has_weights:
                    batch['weights'] = self.xp.asarray(
                        weights_step, dtype=self.xp.float32)
                loss += self._compute_loss(batch,
                                           errors_out=errors_out_step)
                if errors_out is not None:
                    for err, index in zip(errors_out_step, indices):
                        errors_out[index] += err
            loss /= max_epi_len

            # Update stats
            self.average_loss *= self.average_loss_decay
            self.average_loss += \
                (1 - self.average_loss_decay) * float(loss.array)

            self.model.cleargrads()
            loss.backward()
            self.optimizer.update()
        if has_weights:
            self.replay_buffer.update_errors(errors_out)

    def _compute_target_values(self, exp_batch):
        batch_next_state = exp_batch['next_state']

        if self.t % 100 == 0 and self.plot:
            import matplotlib.pyplot as plt
            import numpy as np
            g = 0.9
            trueq = [
                g**6, g**5, g**6, g**6,
                -g, -1, -g, -1,
                g**4, g**3, -g, -1,
                -1, g**2, g, 1]
            sample_state = self.xp.array(np.identity(16), dtype=self.xp.float32)

            if self.head:
                target_next_qout = self.target_model(sample_state, **{'noise': False})
                #print(target_next_qout.sigmas)
                plt.errorbar(target_next_qout.q_values.data[:, 1], -np.arange(16),
                    xerr=target_next_qout.sigmas.data[:, 1], fmt='ok', lw=2)
            else:
                samples = 0
                pts = []

                for i in range(100):
                    target_next_qout = self.target_model(sample_state, **{'noise': False})
                    samples += target_next_qout.q_values
                    pts.append(target_next_qout.q_values.data)
                    #samples.append(target_next_qout.q_values)

                pts2 = []

                for i in range(100):
                    target_next_qout = self.model(sample_state, **{'noise': False})
                    pts2.append(target_next_qout.q_values.data)

                nonoise = self.target_model(sample_state, **{'noise': False, 'target': True}).q_values
                samples /= 100

                plt.title("Q(S, right) @ step: %s" % self.t)
                plt.xlim((-1, 1))

                pts = np.array(pts)[:, :, 1].T
                pts = pts.flatten()

                plt.plot(pts, np.repeat(-np.arange(16), 100), "x", c="blue")

                pts2 = np.array(pts2)[:, :, 1].T
                pts2 = pts2.flatten()
                plt.plot(pts2, np.repeat(-np.arange(16), 100), ".", c="red")

                plt.plot(nonoise[:, 1].data, -np.arange(16), "o", label="no noise")
                plt.plot(samples[:, 1].data, -np.arange(16), "v", label="sample mean")

                nonoise = self.model(sample_state, **{'noise': False, 'target': True}).q_values
                plt.plot(nonoise[:, 1].data, -np.arange(16), "o", label="model mean")


            plt.plot(trueq, -np.arange(16), "^", label="true Q")

            plt.xlim((-1, 1))
            plt.legend()
            #plt.show()
            plt.savefig(self.vis.outdir + "/plots/" + "%06d" % self.t + ".png")
            plt.clf()

        target_next_qout = self.target_model(batch_next_state, **{'noise': False, 'target': not self.noisy_t, 'avg': True})
        #mean = F.mean(self.xp.array(samples), axis=0)

        #print(nonoise)
        #print(samples)

        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        discount = exp_batch['discount']

        if self.head:
            discount = self.gamma**2.0
            mean = batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

            if self.algo == "DQN":
                sigma = discount * target_next_qout.max_sigma
            else:
                acts = exp_batch['next_action']
                sigma = discount * target_next_qout.evaluate_action_sigmas(acts)

            return mean, sigma
        else:
            return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, exp_batch, gamma):
        #print(self.model.q_func.layers[0][0].W[0, 0])

        batch_size = exp_batch['reward'].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch['state']

        qout = self.model(batch_state, **{'noise': False, 'target': not self.noisy_y, 'avg': False})

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))
        if self.head:
            batch_sigma = F.reshape(qout.evaluate_action_sigmas(
                batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            if self.head:
                batch_q_target, batch_sigma_target = self._compute_target_values(exp_batch)
                batch_q_target = F.reshape(batch_q_target, (batch_size, 1))
                batch_sigma_target = F.reshape(batch_sigma_target, (batch_size, 1))
            else:
                batch_q_target = F.reshape(
                    self._compute_target_values(exp_batch, gamma),
                    (batch_size, 1))

        if self.head:
            if hasattr(qout, 'all_sigmas') and qout.all_sigmas is not None:
                targets = [batch_q_target]
                curr = [batch_q]
                for sig in qout.all_sigmas:
                    targets.append(batch_sigma_target)
                    curr.append(F.reshape(F.select_item(sig, batch_actions), (batch_size, 1)))
                return F.concat(curr, axis=1), F.concat(targets, axis=1), qout.sigmas
            else:
                return F.concat([batch_q, batch_sigma], axis=1),\
                    F.concat([batch_q_target, batch_sigma_target], axis=1),\
                    qout.sigmas
        else:
            return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, gamma, errors_out=None):
        """Compute the Q-learning loss for a batch of experiences


        Args:
          experiences (list): see update()'s docstring
          discount (float): Amount by the Q-values should be discounted
        Returns:
          Computed loss from the minibatch of experiences
        """
        if self.head:
            y, t, sigma = self._compute_y_and_t(exp_batch, gamma)
        else:
            y, t = self._compute_y_and_t(exp_batch, gamma)

        entropy_loss = 0

        if self.entropy or self.entropy_coef > 0:
            if self.head:
                entropy_loss = -(F.sum(F.log(F.absolute(sigma)**2.0 + 1e-5)))
            else:
                entropy_loss = -(sum([m.entropy for m in self.entropy]))

            self.last_entropy = entropy_loss
            entropy_loss *= self.entropy_coef# * (1.0 - self.t/5000.0)
            #print("EEEE" + entropy_loss)

        if errors_out is not None:
            del errors_out[:]
            delta = F.sum(abs(y - t), axis=1)
            delta = cuda.to_cpu(delta.array)
            for e in delta:
                errors_out.append(e)

        if 'weights' in exp_batch:
            return compute_weighted_value_loss(
                y, t, exp_batch['weights'],
                clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator) + entropy_loss
        else:
            return compute_value_loss(y, t, clip_delta=self.clip_delta,
                                      batch_accumulator=self.batch_accumulator) + entropy_loss

    def act(self, obs):
        state_index = self.to_state_index(obs)

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                if self.use_table:
                    action_value = self.q_table_mu[state_index, :]
                    q = action_value.max()
                    action = cuda.to_cpu(action_value.argmax())
                else:
                    action_value = self.model(
                        self.batch_states([obs], self.xp, self.phi))#, **{'noise': True, 'act': True, 'avg': False})
                    q = float(action_value.max.data)
                    action = cuda.to_cpu(action_value.greedy_actions.data)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action

    def update_noise_std(self, s):
        noisy = self.model(
            self.batch_states([s], self.xp, self.phi), **{'noise': True, 'avg': True}).q_values
        clean = self.model(
            self.batch_states([s], self.xp, self.phi), **{'noise': False, 'avg': True}).q_values
        noisy, clean = F.softmax(noisy), F.softmax(clean)
        div = -F.sum(clean * F.log(noisy / clean))
        delta = 0.05
        #if div.data <= delta:
        #    self.model.scale_noise_coef(1.01)
        #else:
        #    self.model.scale_noise_coef(1/1.01)

    def plot_values2(self):
        x = np.linspace(-1.3, 0.7, 20)
        y = np.linspace(-0.08, 0.08, 20)
        xv, yv = np.meshgrid(x, y)
        vecs = np.stack([xv.flatten(), yv.flatten()]).T
        if self.nn:
            vals = self.model(self.xp.asarray(vecs.astype(np.float32)))
        import cv2

        if self.gpu >= 0:
            if self.nn:
                arr = self.xp.asnumpy
            else:
                arr = np.asarray
        else:
            arr = np.asarray

        hdivider = np.zeros((128, 30, 3))
        hdivider[:, :, 0] = 0.7

        empty = np.zeros((128, 128, 3))
        empty[:,:,0] = 1

        def normalize(data, min=None, max=None):
            data = data.copy()

            if min is None:
                min = data.min()

            data -= min

            if max is None and data.max() != 0:
                max = data.max()
                data /= max
            else:
                data /= (max-min)

            data = np.clip(data, 0, 1)
            data = data.reshape((self.res, self.res))
            data = cv2.resize(data, (128, 128), interpolation=cv2.INTER_NEAREST)
            img = np.dstack([data] * 3)

            return img

        def get_row(act):
            if self.nn:
                data = arr(vals.q_values.data)[:, act]
                #means = normalize(data, -100, 0)
                means = normalize(data, -10, 0)

                try:
                    data = arr(vals.sigmas.data)[:, act]
                    sigmas = normalize(data, 0)
                except:
                    sigmas = np.zeros((128, 128, 3))
            else:
                means = empty
                sigmas = empty

            counts = self.counts[:,act] / self.counts[:,act].max()
            counts = normalize(counts)

            counts2 = self.counts2[:,act] / self.counts2[:,act].max()
            counts2 = normalize(counts2)

            table_mean = arr(self.q_table_mu[:, act])
            #table_mean = normalize(table_mean, -100, 0)
            table_mean = normalize(table_mean, -10, 0)

            table_var = arr(self.q_table_var[:, act])
            table_var = normalize(table_var, 0)

            canvas = np.hstack([means, hdivider, sigmas, hdivider, counts, hdivider, counts2, hdivider,
                table_mean, hdivider, table_var])

            return canvas

        #print(self.q_table_mu, self.q_table_mu.max())
        row1 = get_row(0)
        row2 = get_row(1)
        row3 = get_row(2)

        header = np.zeros((70, row1.shape[1], 3))
        header[:, :, 0] = 0.7
        header = cv2.putText(header, 'NN_Q_mu       NN_Q_sigma       state_visits       state_visits_recent       table_Q_mu       table_Q_sigma', (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1,1,1), 1)

        bottom = np.zeros((70, row1.shape[1], 3))
        bottom[:, :, 0] = 0.7
        bottom = cv2.putText(bottom, 'step:' + str(self.t) + ' last score:' + self.last_score, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (1,1,1), 2)

        x = np.arange(self.res)
        y = np.arange(20)
        xv, yv = np.meshgrid(x, y)
        xv = xv.astype(np.int32)
        yv = yv.astype(np.int32)



        def get_max(data):
            acts = np.zeros((self.res, self.res, 3))
            #print(xv.flatten(), yv.flatten())
            acts[yv.flatten(), xv.flatten(), arr(data).argmax(axis=1)] = 1
            acts = cv2.resize(acts, (128, 128), interpolation=cv2.INTER_NEAREST)

            return acts

        def get_softmax(data):
            acts = arr(data).reshape((self.res, self.res, 3))
            acts = np.exp(acts) / np.exp(acts).sum(axis=2)[:,:,None]
            acts = cv2.resize(acts, (128, 128), interpolation=cv2.INTER_NEAREST)

            return acts

        divider = np.zeros((128, 30, 3))
        divider[:, :, 0] = 0.7
        if self.nn:
            acts = get_max(vals.q_values.data)
            acts2 = get_max(self.q_table_mu)
            acts_soft = get_softmax(vals.q_values.data)
            acts2_soft = get_softmax(self.q_table_mu)
        else:
            acts2 = get_max(self.q_table_mu)
            acts2_soft = empty#get_softmax(self.q_table_mu)
            acts = empty
            acts_soft = empty

        if self.use_table:
            #print(self.q_table_sigma.max(), self.q_table_sigma.min())
            empty = estimate(self.xp, self.q_table_mu, self.q_table_var*0.1, 10).reshape((20, 20, 3))
            empty = cv2.resize(arr(empty), (128, 128), interpolation=cv2.INTER_NEAREST)
        elif hasattr(vals, 'sigmas') and vals.sigmas is not None:
            if self.table_sigma:
                empty = estimate(self.xp, vals.q_values.data, self.q_table_var, 10).reshape((20, 20, 3))
            else:
                empty = estimate(self.xp, vals.q_values.data, vals.sigmas.data, 10).reshape((20, 20, 3))
            empty = cv2.resize(arr(empty), (128, 128), interpolation=cv2.INTER_NEAREST)

        visits = cv2.resize(self.visited.reshape(self.res, self.res, 3), (128, 128), interpolation=cv2.INTER_NEAREST)


        #print(acts.shape, divider.shape, empty.shape)
        acts = np.hstack([acts, divider, acts_soft, divider, visits, divider, empty, divider,
            acts2, divider, acts2_soft])

        divider = np.zeros((30, row1.shape[1], 3))
        divider[:, :, 0] = 0.7

        if self.nn and hasattr(vals, 'all_sigmas') and vals.all_sigmas is not None:
            all_sigmas = vals.all_sigmas.data[:5]
            all_sigmas = all_sigmas.reshape((all_sigmas.shape[0], 20, 20, all_sigmas.shape[-1]))
            norms = []
            for i, m in enumerate(all_sigmas):
                #norm = get_softmax(m)
                if self.use_gpu:
                    m = self.xp.asnumpy(m)
                norm = normalize(m[:,:,0], 0)
                norms.append(norm)
                norms.append(hdivider)
            norms = np.hstack(norms)
            nc = np.zeros((norms.shape[0], row1.shape[1], 3))
            nc[:norms.shape[0], :norms.shape[1]] = norms

            canvas = np.vstack([header, row1, divider, row2, divider, row3, divider, acts, divider,
            nc, bottom])
        else:
            canvas = np.vstack([header, row1, divider, row2, divider, row3, divider, acts, divider, bottom])

        #ow('test', canvas)
        #cv2.waitKey(1)
        #cv2.imwrite('frames2/%06d.png' % self.t, canvas*255.0)

        #print('writing vid')
        if self.vid is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.vid = cv2.VideoWriter(self.outdir + '/output.avi',fourcc, 20.0, (canvas.shape[1], canvas.shape[0]))

        #try:

        """
        if self.use_gpu:
            xp = self.xp.asnumpy
        else:
            xp = self.xp.asarray

        pltax.scatter([0, 1, 2], xp(self.pi[0]), label="samples")
        pltax.scatter([0, 1, 2], xp(self.pi[1]), label="estimate")
        pltax.legend()
        #except:
        #    print('error')

        pltcanvas.draw()

        width, height = pltfig.get_size_inches() * pltfig.get_dpi()
        #image = np.fromstring(pltcanvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        image = np.array(pltfig.canvas.renderer._renderer)
        c = np.ones_like(image[:,:,:3])

        c += image[:,:,:3]*image[:,:,[-1,-1,-1]]
        #w, h = c.shape[0], c.shape[1]
        c = cv2.resize(c, (128, 128))
        canvas[3*(128+30)+70:3*(128+30)+70+128, 128+30:128+30+128] = c
        """

        self.vid.write((canvas*255.0).astype(np.uint8))

        #cv2.imwrite("results/debug.png", canvas*255.0)
        if self.render:
            cv2.imshow("vis", canvas)
            cv2.waitKey(1)

        #pltax.clear()

    def to_state_index(self, obs):
        ind = 0

        for i, o in enumerate(obs):
            low = self.gym_env.observation_space.low[i]

            if low < -100:
                low = -1
                range = 2
            else:
                range = self.gym_env.observation_space.high[i] - low

            num = int(self.res * (float(o) - low) / range)
            num = max(min(num, self.res-1), 0)
            num *= self.res**i
            ind += num

        return ind

    def act_and_train(self, obs, reward):
        state_index = self.to_state_index(obs)

        if not self.fixed_sigma:
            self.noise_table = self.xp.asarray(np.random.normal(size=(self.n_states, self.n_actions)))

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                if self.use_table:
                    if self.table_sigma:
                        action_value = self.q_table_mu[state_index, :].copy()

                        if self.table_noise:
                            if self.one_sigma:
                                action_value += self.noise_table[0, :] * self.xp.sqrt(self.xp.abs(self.q_table_var[state_index, :]))
                            else:
                                action_value += self.noise_table[state_index, :] * self.xp.sqrt(self.xp.abs(self.q_table_var[state_index, :]))
                        else:
                            #print("a", self.noise_table[vel*20+pos, :])
                            #print("b", self.xp.random.normal(action_value.shape))
                            #action_value += self.noise_table[vel*20+pos, :] * self.q_table_sigma[vel*20+pos, :]
                            action_value += self.xp.random.normal(size=action_value.shape) * self.xp.sqrt(self.xp.abs(self.q_table_var[state_index, :]))

                        q = action_value.max()
                        greedy_action = cuda.to_cpu(action_value.argmax())
                    else:
                        action_value = self.q_table_mu[state_index, :].copy()
                        q = action_value.max()
                        greedy_action = cuda.to_cpu(action_value.argmax())
                else:
                    action_value = self.model(
                        self.batch_states([obs], self.xp, self.phi))
                    q = float(action_value.max.data)

                    if self.head:
                        if self.table_sigma:
                            table_sigma = self.xp.maximum(self.xp.sqrt(self.xp.abs(self.q_table_var[state_index, :])), self.min_sigma)

                            if self.table_noise:
                                if self.one_sigma:
                                    noise = self.noise_table[0, :] * self.xp.sqrt(self.xp.abs(self.q_table_var[state_index, :]))
                                    greedy_action = cuda.to_cpu(action_value.sample_actions_given_noise(noise).data)[0]
                                else:
                                    noise = self.noise_table[state_index, :] * self.xp.sqrt(self.xp.abs(self.q_table_var[state_index, :]))
                                    greedy_action = cuda.to_cpu(action_value.sample_actions_given_noise(noise).data)[0]
                            else:
                                sigma = self.xp.sqrt(self.xp.abs(self.q_table_var[state_index, :]))
                                greedy_action = cuda.to_cpu(action_value.sample_actions_given_sigma(sigma).data)[0]
                        else:
                            #greedy_action = cuda.to_cpu(action_value.greedy_actions.data)[0]
                            greedy_action = cuda.to_cpu(action_value.sample_actions.data)[0]
                    else:
                        greedy_action = cuda.to_cpu(action_value.greedy_actions.data)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        #self.logger.info('a:%s', action)

        #if "car" in self.env:
        self.counts *= 0.9999
        self.counts[state_index, action] += 0.0001
        self.counts2 *= 0.99
        self.counts2[state_index, action] += 0.01
        self.visited[state_index, action] = 1

        self.t += 1

        #if self.t % 50 == 0:
        #    self.update_noise_std(obs)

        if self.t % 100 == 0:
            if self.vis and self.plot and "car" in self.env:
                self.vis.plot_values(len(obs), self)

        if self.video and self.t % 100 == 0 and "car" in self.env:
            self.plot_values2()

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

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

        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def stop_episode_and_train(self, state, reward, done=False):
        """Observe a terminal state and a reward.

        This function must be called once when an episode terminates.
        """

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
        #self.model.reset_noise()
        self.noise_table = self.xp.asarray(np.random.normal(size=(self.n_states, self.n_actions)))
        self.last_state = None
        self.last_action = None
        if isinstance(self.model, Recurrent):
            self.model.reset_state()
        self.replay_buffer.stop_current_episode()

    def get_statistics(self):
        stats = [
            ('average_q', self.average_q),
            ('average_loss', self.average_loss),
            ('n_updates', self.optimizer.t),
        ]

        if self.entropy is not None:
            for i, noise in enumerate(self.entropy):
                s = F.mean(F.absolute(noise.sigma.W)).data
                stats.append(('entropy_W_' + str(i), s))

                #if not noise.nobias:
                #    s = F.mean(F.absolute(noise.sigma.b)).data
                #    stats.append(('entropy_b_' + str(i), s))
            stats.append(('entropy_loss', self.last_entropy))

        xp = self.xp

        def eval_vals(x, name):
            mod = self.model(x)
            vals = mod.q_values.data
            sigs = mod.sigmas.data
            stats.append((name + '_q_mean', xp.mean(vals)))
            stats.append((name + '_q_std', xp.mean(xp.std(vals, axis=1))))

            vals = xp.exp(vals) / xp.exp(vals).sum(axis=1)[:, None]
            ent = -xp.sum(vals * xp.log(vals), axis=1)
            stats.append((name + '_q_ent', xp.mean(ent)))

            stats.append((name + '_q_sigma_mean', xp.mean(sigs)))

        if not self.conv and "car" in self.env:
            try:
                eval_vals(self.xp.asarray([[-1.5, -0.1], [-1.5, 0.1], [1.0, -0.1], [1.0, 0.1]], dtype=self.xp.float32), 'custom')
                eval_vals(self.xp.random.uniform(-5, 5, (32, 2), dtype=self.xp.float32), 'random')
            except:
                self.conv = True

        stats.append(('est_error', self.est_error))
        stats.append(('visited_s', np.sum(np.max(self.visited,1))))
        stats.append(('visited_sa', np.sum(self.visited)))

        return stats
