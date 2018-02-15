from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import copy

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states


def _elementwise_clip(x, x_min, x_max):
    """Elementwise clipping

    Note: chainer.functions.clip supports clipping to constant intervals
    """
    return F.minimum(F.maximum(x, x_min), x_max)


class PPO(agent.AttributeSavingMixin, agent.Agent):
    """Proximal Policy Optimization

    See https://arxiv.org/abs/1707.06347

    Args:
        model (A3CModel): Model to train.  Recurrent models are not supported.
            state s  |->  (pi(s, _), v(s))
        optimizer (chainer.Optimizer): Optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        value_func_coef (float): Weight coefficient for loss of
            value function (0, inf)
        entropy_coef (float): Weight coefficient for entropoy bonus [0, inf)
        update_interval (int): Model update interval in step
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        clip_eps (float): Epsilon for pessimistic clipping of likelihood ratio
            to update policy
        clip_eps_vf (float): Epsilon for pessimistic clipping of value
            to update value function. If it is ``None``, value function is not
            clipped on updates.
        standardize_advantages (bool): Use standardized advantages on updates
        average_v_decay (float): Decay rate of average V, only used for
            recording statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            recording statistics
    """

    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer,
                 gpu=None,
                 gamma=0.99,
                 lambd=0.95,
                 phi=lambda x: x,
                 value_func_coef=1.0,
                 entropy_coef=0.01,
                 update_interval=2048,
                 minibatch_size=64,
                 epochs=10,
                 clip_eps=0.2,
                 clip_eps_vf=None,
                 standardize_advantages=True,
                 average_v_decay=0.999, average_loss_decay=0.99,
                 ):
        self.model = model

        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)

        self.optimizer = optimizer
        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.value_func_coef = value_func_coef
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.clip_eps_vf = clip_eps_vf
        self.standardize_advantages = standardize_advantages

        self.average_v = 0
        self.average_v_decay = average_v_decay
        self.average_loss_policy = 0
        self.average_loss_value_func = 0
        self.average_loss_entropy = 0
        self.average_loss_decay = average_loss_decay

        self.xp = self.model.xp
        self.last_state = None

        self.memory = []
        self.last_episode = []

    def _act(self, state):
        xp = self.xp
        with chainer.using_config('train', False):
            b_state = batch_states([state], xp, self.phi)
            with chainer.no_backprop_mode():
                action_distrib, v = self.model(b_state)
                action = action_distrib.sample()
            return cuda.to_cpu(action.data)[0], cuda.to_cpu(v.data)[0]

    def _train(self):
        if len(self.memory) + len(self.last_episode) >= self.update_interval:
            self._flush_last_episode()
            self.update()
            self.memory = []

    def _flush_last_episode(self):
        if self.last_episode:
            self._compute_teacher()
            self.memory.extend(self.last_episode)
            self.last_episode = []

    def _compute_teacher(self):
        """Estimate state values and advantages of self.last_episode

        TD(lambda) estimation
        """

        adv = 0.0
        for transition in reversed(self.last_episode):
            td_err = (
                transition['reward']
                + (self.gamma * transition['nonterminal']
                   * transition['next_v_pred'])
                - transition['v_pred']
                )
            adv = td_err + self.gamma * self.lambd * adv
            transition['adv'] = adv
            transition['v_teacher'] = adv + transition['v_pred']

    def _lossfun(self,
                 distribs, vs_pred, log_probs,
                 vs_pred_old, target_log_probs,
                 advs, vs_teacher):
        prob_ratio = F.exp(log_probs - target_log_probs)
        ent = distribs.entropy

        prob_ratio = F.expand_dims(prob_ratio, axis=-1)
        loss_policy = - F.mean(F.minimum(
            prob_ratio * advs,
            F.clip(prob_ratio, 1-self.clip_eps, 1+self.clip_eps) * advs))

        if self.clip_eps_vf is None:
            loss_value_func = F.mean_squared_error(vs_pred, vs_teacher)
        else:
            loss_value_func = F.mean(F.maximum(
                F.square(vs_pred - vs_teacher),
                F.square(_elementwise_clip(vs_pred,
                                           vs_pred_old - self.clip_eps_vf,
                                           vs_pred_old + self.clip_eps_vf)
                         - vs_teacher)
                ))

        loss_entropy = -F.mean(ent)

        # Update stats
        self.average_loss_policy += (
            (1 - self.average_loss_decay) *
            (cuda.to_cpu(loss_policy.data) - self.average_loss_policy))
        self.average_loss_value_func += (
            (1 - self.average_loss_decay) *
            (cuda.to_cpu(loss_value_func.data) - self.average_loss_value_func))
        self.average_loss_entropy += (
            (1 - self.average_loss_decay) *
            (cuda.to_cpu(loss_entropy.data) - self.average_loss_entropy))

        return (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * loss_entropy
            )

    def update(self):
        xp = self.xp

        if self.standardize_advantages:
            all_advs = xp.array([b['adv'] for b in self.memory])
            mean_advs = xp.mean(all_advs)
            std_advs = xp.std(all_advs)

        target_model = copy.deepcopy(self.model)
        dataset_iter = chainer.iterators.SerialIterator(
            self.memory, self.minibatch_size)

        dataset_iter.reset()
        while dataset_iter.epoch < self.epochs:
            batch = dataset_iter.__next__()
            states = batch_states([b['state'] for b in batch], xp, self.phi)
            actions = xp.array([b['action'] for b in batch])
            distribs, vs_pred = self.model(states)
            with chainer.no_backprop_mode():
                target_distribs, _ = target_model(states)

            advs = xp.array([b['adv'] for b in batch], dtype=xp.float32)
            if self.standardize_advantages:
                advs = (advs - mean_advs) / std_advs

            self.optimizer.update(
                self._lossfun,
                distribs, vs_pred, distribs.log_prob(actions),
                vs_pred_old=xp.array(
                    [b['v_pred'] for b in batch], dtype=xp.float32),
                target_log_probs=target_distribs.log_prob(actions),
                advs=advs,
                vs_teacher=xp.array(
                    [b['v_teacher'] for b in batch], dtype=xp.float32),
                )

    def act_and_train(self, obs, reward):
        if hasattr(self.model, 'obs_filter'):
            xp = self.xp
            b_state = batch_states([obs], xp, self.phi)
            self.model.obs_filter.experience(b_state)

        action, v = self._act(obs)

        # Update stats
        self.average_v += (
            (1 - self.average_v_decay) *
            (v[0] - self.average_v))

        if self.last_state is not None:
            self.last_episode.append({
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'v_pred': self.last_v,
                'next_state': obs,
                'next_v_pred': v,
                'nonterminal': 1.0})
        self.last_state = obs
        self.last_action = action
        self.last_v = v

        self._train()
        return action

    def act(self, obs):
        action, v = self._act(obs)

        # Update stats
        self.average_v += (
            (1 - self.average_v_decay) *
            (v[0] - self.average_v))

        return action

    def stop_episode_and_train(self, state, reward, done=False):
        _, v = self._act(state)

        assert self.last_state is not None
        self.last_episode.append({
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'v_pred': self.last_v,
            'next_state': state,
            'next_v_pred': v,
            'nonterminal': 0.0 if done else 1.0})

        self.last_state = None
        del self.last_action
        del self.last_v

        self._flush_last_episode()
        self.stop_episode()

    def stop_episode(self):
        pass

    def get_statistics(self):
        return [
            ('average_v', self.average_v),
            ('average_loss_policy', self.average_loss_policy),
            ('average_loss_value_func', self.average_loss_value_func),
            ('average_loss_entropy', self.average_loss_entropy),
            ]
