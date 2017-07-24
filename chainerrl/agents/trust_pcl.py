import chainer
import chainer.functions as F
from collections import deque
import copy
import numpy as np

import chainerrl
from chainerrl.agents.pcl import PCL
from chainerrl.misc.copy_param import soft_copy_param
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_kept
from chainerrl.recurrent import state_reset
from chainerrl.replay_buffer import batch_experiences


def estimate_policy_divergence(returns, lambd):
    """calculate KL(pi*||tilde{pi}) from sampled returns

    pi* is the optimal policy under the penalty coefficient lambd
    args:
        returns (array of floats): regarded as sampled returns
            from trajectories by the policy tilde{pi}
        lambd (float): param
    """

    returns = np.array([R / lambd for R in returns])
    max_return = np.amax(returns)
    logZ = np.log(np.mean(np.exp(returns - max_return))) + max_return
    return np.mean(returns * np.exp(returns - logZ)) - logZ


def binary_search(increasing_func, low, high, precision):
    while True:
        mid = (low + high) / 2
        if high - low < precision:
            return mid
        y = increasing_func(mid)
        if y < 0:
            low = mid
        else:
            high = mid


class TrustPCL(PCL):
    """Trust-PCL

    See https://arxiv.org/abs/1707.01891

    Args:
        alpha (float): parameter for lagged geometric mean of model that
            will be used for trust region
        epsilon (float): desired divergence KL(policy||target_policy)
            that tunes lambda, the penalty coefficient of trust region

        for other arguments see PCL's arguments
    """
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha', 0.95)
        self.epsilon = kwargs.pop('epsilon', 0.002)
        super().__init__(*args, **kwargs)
        self.target_model = copy.deepcopy(self.model)
        self.lambd = None
        self.returns = deque(maxlen=100)  # 100 \times alpha \approx 1
        self.current_return = 0.0

    def act_and_train(self, state, reward):
        if reward:
            self.current_return += reward

        # Do it before self.t is incremented
        statevar = self.batch_states([state], self.xp, self.phi)
        target_action_distrib, _ = self.target_model(statevar)
        self.past_target_action_distrib[self.t] = target_action_distrib

        return super().act_and_train(state, reward)

    def stop_episode_and_train(self, state, reward, done=False):
        if reward:
            self.current_return += reward
        self.returns.append(self.current_return)
        if self.lambd is not None:
            self.update_lambd()
        self.current_return = 0.0

        if isinstance(self.model, Recurrent):
            self.model.reset_state()
        if isinstance(self.target_model, Recurrent):
            self.model.reset_state()

        super().stop_episode_and_train(state, reward, done)

    def update_lambd(self):
        self.lambd = binary_search(
            (lambda lambd:
             self.epsilon - estimate_policy_divergence(self.returns, lambd)),
            low=0.0, high=1000.0, precision=0.1)

    def init_history_data_for_online_update(self):
        super().init_history_data_for_online_update()
        self.past_target_action_distrib = {}

    def compute_loss(self, t_start, t_stop, rewards, values,
                     next_values, log_probs, target_log_probs):

        if self.lambd is None:
            self.update_lambd()

        seq_len = t_stop - t_start
        assert len(rewards) == seq_len
        assert len(values) == seq_len
        assert len(next_values) == seq_len
        assert len(log_probs) == seq_len

        pi_losses = []
        v_losses = []
        for t in range(t_start, t_stop):
            d = min(t_stop - t, self.rollout_len)
            # Discounted sum of immediate rewards
            R_seq = sum(self.gamma ** i * rewards[t + i] for i in range(d))
            # Discounted sum of log likelihoods
            G_abs = chainerrl.functions.weighted_sum_arrays(
                xs=[log_probs[t + i] for i in range(d)],
                weights=[self.gamma ** i for i in range(d)])
            G_rel = chainerrl.functions.weighted_sum_arrays(
                xs=[log_probs[t + i] - target_log_probs[t + i]
                    for i in range(d)],
                weights=[self.gamma ** i for i in range(d)])
            G = F.expand_dims(self.tau * G_abs + self.lambd * G_rel, -1)
            last_v = next_values[t + d - 1]
            if not self.backprop_future_values:
                last_v = chainer.Variable(last_v.data)

            # C_pi only backprop through pi
            C_pi = (- values[t].data +
                    self.gamma ** d * last_v.data +
                    R_seq - G)

            # C_v only backprop through v
            C_v = (- values[t] +
                   self.gamma ** d * last_v +
                   R_seq - G.data)

            pi_losses.append(C_pi ** 2)
            v_losses.append(C_v ** 2)

        pi_loss = chainerrl.functions.sum_arrays(pi_losses) / 2
        v_loss = chainerrl.functions.sum_arrays(v_losses) / 2

        # Re-scale pi loss so that it is independent from tau
        pi_loss /= self.tau

        pi_loss *= self.pi_loss_coef
        v_loss *= self.v_loss_coef

        if self.normalize_loss_by_steps:
            pi_loss /= seq_len
            v_loss /= seq_len

        if self.process_idx == 0:
            self.logger.debug('pi_loss:%s v_loss:%s',
                              pi_loss.data, v_loss.data)

        return pi_loss + F.reshape(v_loss, pi_loss.data.shape)

    def update(self, loss):
        super().update(loss)
        soft_copy_param(self.target_model, self.model, self.alpha)

    def update_from_replay(self):

        if self.replay_buffer is None:
            return

        if len(self.replay_buffer) < self.replay_start_size:
            return

        if self.process_idx == 0:
            self.logger.debug('update_from_replay')

        episodes = self.replay_buffer.sample_episodes(
            self.batchsize, max_len=self.t_max)
        if isinstance(episodes, tuple):
            # Prioritized replay
            episodes, weights = episodes
        else:
            weights = [1] * len(episodes)
        sorted_episodes = list(reversed(sorted(episodes, key=len)))
        max_epi_len = len(sorted_episodes[0])

        self.target_model.reset_state()
        with state_reset(self.model):
            # Batch computation of multiple episodes
            rewards = {}
            values = {}
            next_values = {}
            log_probs = {}
            target_log_probs = {}
            next_action_distrib = None
            next_v = None
            for t in range(max_epi_len):
                transitions = []
                for ep in sorted_episodes:
                    if len(ep) <= t:
                        break
                    transitions.append(ep[t])
                batch = batch_experiences(transitions,
                                          xp=self.xp,
                                          phi=self.phi,
                                          batch_states=self.batch_states)
                batchsize = batch['action'].shape[0]
                if next_action_distrib is not None:
                    action_distrib = next_action_distrib[0:batchsize]
                    v = next_v[0:batchsize]
                else:
                    action_distrib, v = self.model(batch['state'])
                target_action_distrib, _ = self.target_model(batch['state'])
                next_action_distrib, next_v = self.model(batch['next_state'])
                values[t] = v
                next_values[t] = next_v * \
                    (1 - batch['is_state_terminal'].reshape(next_v.shape))
                rewards[t] = chainer.cuda.to_cpu(batch['reward'])
                log_probs[t] = action_distrib.log_prob(batch['action'])
                target_log_probs[t] = \
                    target_action_distrib.log_prob(batch['action'])
            # Loss is computed one by one episode
            losses = []
            for i, ep in enumerate(sorted_episodes):
                e_values = {}
                e_next_values = {}
                e_rewards = {}
                e_log_probs = {}
                e_target_log_probs = {}
                for t in range(len(ep)):
                    assert values[t].shape[0] > i
                    assert next_values[t].shape[0] > i
                    assert rewards[t].shape[0] > i
                    assert log_probs[t].shape[0] > i
                    assert target_log_probs[t].shape[0] > i
                    e_values[t] = values[t][i:i + 1]
                    e_next_values[t] = next_values[t][i:i + 1]
                    e_rewards[t] = float(rewards[t][i:i + 1])
                    e_log_probs[t] = log_probs[t][i:i + 1]
                    e_target_log_probs[t] = target_log_probs[t][i:i + 1]
                losses.append(self.compute_loss(
                    t_start=0,
                    t_stop=len(ep),
                    rewards=e_rewards,
                    values=e_values,
                    next_values=e_next_values,
                    log_probs=e_log_probs,
                    target_log_probs=e_target_log_probs))
            loss = chainerrl.functions.weighted_sum_arrays(
                losses, weights) / self.batchsize
            self.update(loss)

    def update_on_policy(self, statevar):
        assert self.t_start < self.t

        if not self.disable_online_update:
            next_values = {}
            for t in range(self.t_start + 1, self.t):
                next_values[t - 1] = self.past_values[t]
            if statevar is None:
                next_values[self.t - 1] = chainer.Variable(
                    self.xp.zeros_like(self.past_values[self.t - 1].data))
            else:
                with state_kept(self.model):
                    _, v = self.model(statevar)
                next_values[self.t - 1] = v
            log_probs = {t: self.past_action_distrib[t].log_prob(
                self.xp.asarray(self.xp.expand_dims(a, 0)))
                for t, a in self.past_actions.items()}
            target_log_probs = {t: self.past_target_action_distrib[t].log_prob(
                self.xp.asarray(self.xp.expand_dims(a, 0)))
                for t, a in self.past_actions.items()}
            self.online_batch_losses.append(self.compute_loss(
                t_start=self.t_start, t_stop=self.t,
                rewards=self.past_rewards,
                values=self.past_values,
                next_values=next_values,
                log_probs=log_probs,
                target_log_probs=target_log_probs))
            if len(self.online_batch_losses) == self.batchsize:
                loss = chainerrl.functions.sum_arrays(
                    self.online_batch_losses) / self.batchsize
                self.update(loss)
                self.online_batch_losses = []

        self.init_history_data_for_online_update()
