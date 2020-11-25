import collections
import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np

from chainerrl.agent import AttributeSavingMixin
from chainerrl.agent import BatchAgent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


class TemperatureHolder(chainer.Link):
    """Link that holds a temperature as a learnable value.

    Args:
        initial_log_temperature (float): Initial value of log(temperature).
    """

    def __init__(self, initial_log_temperature=0):
        super().__init__()
        with self.init_scope():
            self.log_temperature = chainer.Parameter(
                np.array(initial_log_temperature, dtype=np.float32))

    def __call__(self):
        """Return a temperature as a chainer.Variable."""
        return F.exp(self.log_temperature)


class SoftActorCritic(AttributeSavingMixin, BatchAgent):
    """Soft Actor-Critic (SAC).

    See https://arxiv.org/abs/1812.05905

    Args:
        policy (Policy): Policy.
        q_func1 (Link): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Link): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        soft_update_tau (float): Tau of soft target update.
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        initial_temperature (float): Initial temperature value. If
            `entropy_target` is set to None, the temperature is fixed to it.
        entropy_target (float or None): If set to a float, the temperature is
            adjusted during training to match the policy's entropy to it.
        temperature_optimizer (Optimizer or None): Optimizer used to optimize
            the temperature. If set to None, Adam with default hyperparameters
            is used.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
    """

    saved_attributes = (
        'policy',
        'q_func1',
        'q_func2',
        'target_q_func1',
        'target_q_func2',
        'policy_optimizer',
        'q_func1_optimizer',
        'q_func2_optimizer',
        'temperature_holder',
        'temperature_optimizer',
    )

    def __init__(
            self,
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            replay_buffer,
            gamma,
            gpu=None,
            replay_start_size=10000,
            minibatch_size=100,
            update_interval=1,
            phi=lambda x: x,
            soft_update_tau=5e-3,
            logger=getLogger(__name__),
            batch_states=batch_states,
            burnin_action_func=None,
            initial_temperature=1.,
            entropy_target=None,
            temperature_optimizer=None,
            act_deterministically=True,
    ):

        self.policy = policy
        self.q_func1 = q_func1
        self.q_func2 = q_func2

        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.policy.to_gpu(device=gpu)
            self.q_func1.to_gpu(device=gpu)
            self.q_func2.to_gpu(device=gpu)

        self.xp = self.policy.xp
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.policy_optimizer = policy_optimizer
        self.q_func1_optimizer = q_func1_optimizer
        self.q_func2_optimizer = q_func2_optimizer
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            n_times_update=1,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
            episodic_update=False,
        )
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func
        self.initial_temperature = initial_temperature
        self.entropy_target = entropy_target
        if self.entropy_target is not None:
            self.temperature_holder = TemperatureHolder(
                initial_log_temperature=np.log(initial_temperature))
            if temperature_optimizer is not None:
                self.temperature_optimizer = temperature_optimizer
            else:
                self.temperature_optimizer = chainer.optimizers.Adam()
            self.temperature_optimizer.setup(self.temperature_holder)
            if gpu is not None and gpu >= 0:
                self.temperature_holder.to_gpu(device=gpu)
        else:
            self.temperature_holder = None
            self.temperature_optimizer = None
        self.act_deterministically = act_deterministically

        self.t = 0
        self.last_state = None
        self.last_action = None

        # Target model
        self.target_q_func1 = copy.deepcopy(self.q_func1)
        self.target_q_func2 = copy.deepcopy(self.q_func2)

        # Statistics
        self.q1_record = collections.deque(maxlen=1000)
        self.q2_record = collections.deque(maxlen=1000)
        self.entropy_record = collections.deque(maxlen=1000)
        self.q_func1_loss_record = collections.deque(maxlen=100)
        self.q_func2_loss_record = collections.deque(maxlen=100)

    @property
    def temperature(self):
        if self.entropy_target is None:
            return self.initial_temperature
        else:
            with chainer.no_backprop_mode():
                return float(self.temperature_holder().array)

    def sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.q_func1,
            dst=self.target_q_func1,
            method='soft',
            tau=self.soft_update_tau,
        )
        synchronize_parameters(
            src=self.q_func2,
            dst=self.target_q_func2,
            method='soft',
            tau=self.soft_update_tau,
        )

    def update_q_func(self, batch):
        """Compute loss for a given Q-function."""

        batch_next_state = batch['next_state']
        batch_rewards = batch['reward']
        batch_terminal = batch['is_state_terminal']
        batch_state = batch['state']
        batch_actions = batch['action']
        batch_discount = batch['discount']

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            next_action_distrib = self.policy(batch_next_state)
            next_actions, next_log_prob =\
                next_action_distrib.sample_with_log_prob()
            next_q1 = self.target_q_func1(batch_next_state, next_actions)
            next_q2 = self.target_q_func2(batch_next_state, next_actions)
            next_q = F.minimum(next_q1, next_q2)
            entropy_term = self.temperature * next_log_prob[..., None]
            assert next_q.shape == entropy_term.shape

            target_q = batch_rewards + batch_discount * \
                (1.0 - batch_terminal) * F.flatten(next_q - entropy_term)

        predict_q1 = F.flatten(self.q_func1(batch_state, batch_actions))
        predict_q2 = F.flatten(self.q_func2(batch_state, batch_actions))

        loss1 = 0.5 * F.mean_squared_error(target_q, predict_q1)
        loss2 = 0.5 * F.mean_squared_error(target_q, predict_q2)

        # Update stats
        self.q1_record.extend(cuda.to_cpu(predict_q1.array))
        self.q2_record.extend(cuda.to_cpu(predict_q2.array))
        self.q_func1_loss_record.append(float(loss1.array))
        self.q_func2_loss_record.append(float(loss2.array))

        self.q_func1_optimizer.update(lambda: loss1)
        self.q_func2_optimizer.update(lambda: loss2)

    def update_temperature(self, log_prob):
        assert not isinstance(log_prob, chainer.Variable)
        loss = -F.mean(
            F.broadcast_to(self.temperature_holder(), log_prob.shape)
            * (log_prob + self.entropy_target))
        self.temperature_optimizer.update(lambda: loss)

    def update_policy_and_temperature(self, batch):
        """Compute loss for actor."""

        batch_state = batch['state']

        action_distrib = self.policy(batch_state)
        actions, log_prob = action_distrib.sample_with_log_prob()
        q1 = self.q_func1(batch_state, actions)
        q2 = self.q_func2(batch_state, actions)
        q = F.minimum(q1, q2)

        entropy_term = self.temperature * log_prob[..., None]
        assert q.shape == entropy_term.shape
        loss = F.mean(entropy_term - q)

        self.policy_optimizer.update(lambda: loss)

        if self.entropy_target is not None:
            self.update_temperature(log_prob.array)

        # Record entropy
        with chainer.no_backprop_mode():
            try:
                self.entropy_record.extend(
                    cuda.to_cpu(action_distrib.entropy.array))
            except NotImplementedError:
                # Record - log p(x) instead
                self.entropy_record.extend(
                    cuda.to_cpu(-log_prob.array))

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""

        batch = batch_experiences(experiences, self.xp, self.phi, self.gamma)
        self.update_q_func(batch)
        self.update_policy_and_temperature(batch)
        self.sync_target_network()

    def batch_select_greedy_action(self, batch_obs, deterministic=False):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
            if deterministic:
                batch_action = self.policy(batch_xs).most_probable.array
            else:
                batch_action = self.policy(batch_xs).sample().array
        return list(cuda.to_cpu(batch_action))

    def select_greedy_action(self, obs, deterministic=False):
        return self.batch_select_greedy_action(
            [obs], deterministic=deterministic)[0]

    def act_and_train(self, obs, reward):

        self.logger.debug('t:%s r:%s', self.t, reward)

        if (self.burnin_action_func is not None
                and self.policy_optimizer.t == 0):
            action = self.burnin_action_func()
        else:
            action = self.select_greedy_action(obs)
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

        return self.last_action

    def act(self, obs):
        return self.select_greedy_action(
            obs, deterministic=self.act_deterministically)

    def batch_act(self, batch_obs):
        return self.batch_select_greedy_action(
            batch_obs, deterministic=self.act_deterministically)

    def batch_act_and_train(self, batch_obs):
        """Select a batch of actions for training.

        Args:
            batch_obs (Sequence of ~object): Observations.

        Returns:
            Sequence of ~object: Actions.
        """

        if (self.burnin_action_func is not None
                and self.policy_optimizer.t == 0):
            batch_action = [self.burnin_action_func()
                            for _ in range(len(batch_obs))]
        else:
            batch_action = self.batch_select_greedy_action(batch_obs)

        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def batch_observe_and_train(
            self, batch_obs, batch_reward, batch_done, batch_reset):
        for i in range(len(batch_obs)):
            self.t += 1
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                self.replay_buffer.append(
                    state=self.batch_last_obs[i],
                    action=self.batch_last_action[i],
                    reward=batch_reward[i],
                    next_state=batch_obs[i],
                    next_action=None,
                    is_state_terminal=batch_done[i],
                    env_id=i,
                )
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

    def batch_observe(self, batch_obs, batch_reward,
                      batch_done, batch_reset):
        pass

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
        return [
            ('average_q1', _mean_or_nan(self.q1_record)),
            ('average_q2', _mean_or_nan(self.q2_record)),
            ('average_q_func1_loss', _mean_or_nan(self.q_func1_loss_record)),
            ('average_q_func2_loss', _mean_or_nan(self.q_func2_loss_record)),
            ('n_updates', self.policy_optimizer.t),
            ('average_entropy', _mean_or_nan(self.entropy_record)),
            ('temperature', self.temperature),
        ]
