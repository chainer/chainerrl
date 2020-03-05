import collections
import itertools
from logging import getLogger
import random

import chainer
import chainer.functions as F
import numpy as np

import chainerrl
from chainerrl import agent
from chainerrl.agents.ppo import _compute_explained_variance
from chainerrl.agents.ppo import _make_dataset
from chainerrl.agents.ppo import _make_dataset_recurrent
from chainerrl.agents.ppo import _yield_subset_of_sequences_with_fixed_number_of_items  # NOQA
from chainerrl.misc.batch_states import batch_states


def _get_ordered_params(link):
    """Get a list of parameters sorted by parameter names."""
    name_param_pairs = list(link.namedparams())
    ordered_name_param_pairs = sorted(name_param_pairs, key=lambda x: x[0])
    return [x[1] for x in ordered_name_param_pairs]


def _flatten_and_concat_variables(vs):
    """Flatten and concat variables to make a single flat vector variable."""
    return F.concat([F.flatten(v) for v in vs], axis=0)


def _as_ndarray(x):
    """chainer.Variable or ndarray -> ndarray."""
    if isinstance(x, chainer.Variable):
        return x.array
    else:
        return x


def _flatten_and_concat_ndarrays(vs):
    """Flatten and concat variables to make a single flat vector ndarray."""
    xp = chainer.cuda.get_array_module(vs[0])
    return xp.concatenate([_as_ndarray(v).ravel() for v in vs], axis=0)


def _split_and_reshape_to_ndarrays(flat_v, sizes, shapes):
    """Split and reshape a single flat vector to make a list of ndarrays."""
    xp = chainer.cuda.get_array_module(flat_v)
    sections = np.cumsum(sizes)
    vs = xp.split(flat_v, sections)
    return [v.reshape(shape) for v, shape in zip(vs, shapes)]


def _replace_params_data(params, new_params_data):
    """Replace data of params with new data."""
    for param, new_param_data in zip(params, new_params_data):
        assert param.shape == new_param_data.shape
        param.array[:] = new_param_data


def _hessian_vector_product(flat_grads, params, vec):
    """Compute hessian vector product efficiently by backprop."""
    grads = chainer.grad([F.sum(flat_grads * vec)], params)
    assert all(grad is not None for grad in grads),\
        "The Hessian-vector product contains None."
    grads_data = [grad.array for grad in grads]
    return _flatten_and_concat_ndarrays(grads_data)


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def _find_old_style_function(outputs):
    """Find old-style functions in the computational graph."""
    found = []
    for v in outputs:
        assert isinstance(v, (chainer.Variable, chainer.variable.VariableNode))
        if v.creator is None:
            continue
        if isinstance(v.creator, chainer.Function):
            found.append(v.creator)
        else:
            assert isinstance(v.creator, chainer.FunctionNode)
        found.extend(_find_old_style_function(v.creator.inputs))
    return found


class TRPO(agent.AttributeSavingMixin, agent.Agent):
    """Trust Region Policy Optimization.

    A given stochastic policy is optimized by the TRPO algorithm. A given
    value function is also trained to predict by the TD(lambda) algorithm and
    used for Generalized Advantage Estimation (GAE).

    Since the policy is optimized via the conjugate gradient method and line
    search while the value function is optimized via SGD, these two models
    should be separate.

    Since TRPO requires second-order derivatives to compute Hessian-vector
    products, Chainer v3.0.0 or newer is required. In addition, your policy
    must contain only functions that support second-order derivatives.

    See https://arxiv.org/abs/1502.05477 for TRPO.
    See https://arxiv.org/abs/1506.02438 for GAE.

    Args:
        policy (Policy): Stochastic policy. Its forward computation must
            contain only functions that support second-order derivatives.
            Recurrent models are not supported.
        vf (ValueFunction): Value function. Recurrent models are not supported.
        vf_optimizer (chainer.Optimizer): Optimizer for the value function.
        obs_normalizer (chainerrl.links.EmpiricalNormalization or None):
            If set to chainerrl.links.EmpiricalNormalization, it is used to
            normalize observations based on the empirical mean and standard
            deviation of observations. These statistics are updated after
            computing advantages and target values and before updating the
            policy and the value function.
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        entropy_coef (float): Weight coefficient for entropoy bonus [0, inf)
        update_interval (int): Interval steps of TRPO iterations. Every after
            this amount of steps, this agent updates the policy and the value
            function using data from these steps.
        vf_epochs (int): Number of epochs for which the value function is
            trained on each TRPO iteration.
        vf_batch_size (int): Batch size of SGD for the value function.
        standardize_advantages (bool): Use standardized advantages on updates
        line_search_max_backtrack (int): Maximum number of backtracking in line
            search to tune step sizes of policy updates.
        conjugate_gradient_max_iter (int): Maximum number of iterations in
            the conjugate gradient method.
        conjugate_gradient_damping (float): Damping factor used in the
            conjugate gradient method.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        kl_stats_window (int): Window size used to compute statistics
            of KL divergence between old and new policies.
        policy_step_size_stats_window (int): Window size used to compute
            statistics of step sizes of policy updates.

    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated before the value function is updated.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated on act_and_train.
        average_kl: Average of KL divergence between old and new policies.
            It's updated after the policy is updated.
        average_policy_step_size: Average of step sizes of policy updates
            It's updated after the policy is updated.
    """

    saved_attributes = ['policy', 'vf', 'vf_optimizer', 'obs_normalizer']

    def __init__(self,
                 policy,
                 vf,
                 vf_optimizer,
                 obs_normalizer=None,
                 gamma=0.99,
                 lambd=0.95,
                 phi=lambda x: x,
                 entropy_coef=0.01,
                 update_interval=2048,
                 max_kl=0.01,
                 vf_epochs=3,
                 vf_batch_size=64,
                 standardize_advantages=True,
                 batch_states=batch_states,
                 recurrent=False,
                 max_recurrent_sequence_len=None,
                 line_search_max_backtrack=10,
                 conjugate_gradient_max_iter=10,
                 conjugate_gradient_damping=1e-2,
                 act_deterministically=False,
                 value_stats_window=1000,
                 entropy_stats_window=1000,
                 kl_stats_window=100,
                 policy_step_size_stats_window=100,
                 logger=getLogger(__name__),
                 ):

        self.policy = policy
        self.vf = vf
        assert policy.xp is vf.xp, 'policy and vf must be on the same device'
        if recurrent:
            self.model = chainerrl.links.StatelessRecurrentBranched(policy, vf)
        else:
            self.model = chainerrl.links.Branched(policy, vf)
        if policy.xp is not np:
            if hasattr(policy, 'device'):
                # Link.device is available only from chainer v6
                self.model.to_device(policy.device)
            else:
                self.model.to_gpu(device=policy._device_id)
        self.vf_optimizer = vf_optimizer
        self.obs_normalizer = obs_normalizer
        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.max_kl = max_kl
        self.vf_epochs = vf_epochs
        self.vf_batch_size = vf_batch_size
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states
        self.recurrent = recurrent
        self.max_recurrent_sequence_len = max_recurrent_sequence_len
        self.line_search_max_backtrack = line_search_max_backtrack
        self.conjugate_gradient_max_iter = conjugate_gradient_max_iter
        self.conjugate_gradient_damping = conjugate_gradient_damping
        self.act_deterministically = act_deterministically
        self.logger = logger

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.kl_record = collections.deque(maxlen=kl_stats_window)
        self.policy_step_size_record = collections.deque(
            maxlen=policy_step_size_stats_window)
        self.explained_variance = np.nan

        assert self.policy.xp is self.vf.xp,\
            'policy and vf should be in the same device.'
        if self.obs_normalizer is not None:
            assert self.policy.xp is self.obs_normalizer.xp,\
                'policy and obs_normalizer should be in the same device.'
        self.xp = self.policy.xp
        self.last_state = None
        self.last_action = None

        # Contains episodes used for next update iteration
        self.memory = []
        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []

        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

    def _initialize_batch_variables(self, num_envs):
        self.batch_last_episode = [[] for _ in range(num_envs)]
        self.batch_last_state = [None] * num_envs
        self.batch_last_action = [None] * num_envs

    def _update_if_dataset_is_ready(self):
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (0 if self.batch_last_episode is None else sum(
                len(episode) for episode in self.batch_last_episode)))
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                dataset = _make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                )
                self._update_recurrent(dataset)
            else:
                dataset = _make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = _compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory)))
            self.memory = []

    def _flush_last_episode(self):
        if self.last_episode:
            self.memory.append(self.last_episode)
            self.last_episode = []
        if self.batch_last_episode:
            for i, episode in enumerate(self.batch_last_episode):
                if episode:
                    self.memory.append(episode)
                    self.batch_last_episode[i] = []

    def _update(self, dataset):
        """Update both the policy and the value function."""

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)
        self._update_policy(dataset)
        self._update_vf(dataset)

    def _update_recurrent(self, dataset):
        """Update both the policy and the value function."""

        flat_dataset = list(itertools.chain.from_iterable(dataset))
        if self.obs_normalizer:
            self._update_obs_normalizer(flat_dataset)

        self._update_policy_recurrent(dataset)
        self._update_vf_recurrent(dataset)

    def _update_vf_recurrent(self, dataset):

        for epoch in range(self.vf_epochs):
            random.shuffle(dataset)
            for minibatch in _yield_subset_of_sequences_with_fixed_number_of_items(  # NOQA
                    dataset, self.vf_batch_size):
                self._update_vf_once_recurrent(minibatch)

    def _update_vf_once_recurrent(self, episodes):

        xp = self.model.xp
        flat_transitions = list(itertools.chain.from_iterable(episodes))

        # Prepare data for a recurrent model
        seqs_states = []
        for ep in episodes:
            states = self.batch_states(
                [transition['state'] for transition in ep], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            seqs_states.append(states)

        flat_vs_teacher = xp.array(
            [[transition['v_teacher']] for transition in flat_transitions],
            dtype=np.float32)

        with chainer.using_config('train', False),\
                chainer.no_backprop_mode():
            vf_rs = self.vf.concatenate_recurrent_states(
                [ep[0]['recurrent_state'][1] for ep in episodes])

        flat_vs_pred, _ = self.vf.n_step_forward(
            seqs_states, recurrent_state=vf_rs, output_mode='concat')

        vf_loss = F.mean_squared_error(flat_vs_pred, flat_vs_teacher)
        self.vf_optimizer.update(lambda: vf_loss)

    def _update_obs_normalizer(self, dataset):
        assert self.obs_normalizer
        states = batch_states(
            [b['state'] for b in dataset], self.obs_normalizer.xp, self.phi)
        self.obs_normalizer.experience(states)

    def _update_vf(self, dataset):
        """Update the value function using a given dataset.

        The value function is updated via SGD to minimize TD(lambda) errors.
        """

        xp = self.vf.xp

        assert 'state' in dataset[0]
        assert 'v_teacher' in dataset[0]

        dataset_iter = chainer.iterators.SerialIterator(
            dataset, self.vf_batch_size)

        while dataset_iter.epoch < self.vf_epochs:
            batch = dataset_iter.__next__()
            states = batch_states([b['state'] for b in batch], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            vs_teacher = xp.array(
                [b['v_teacher'] for b in batch], dtype=xp.float32)
            vs_pred = self.vf(states)
            vf_loss = F.mean_squared_error(vs_pred, vs_teacher[..., None])
            self.vf_optimizer.update(lambda: vf_loss)

    def _compute_gain(self, log_prob, log_prob_old, entropy, advs):
        """Compute a gain to maximize."""
        prob_ratio = F.exp(log_prob - log_prob_old)
        mean_entropy = F.mean(entropy)
        surrogate_gain = F.mean(prob_ratio * advs)
        return surrogate_gain + self.entropy_coef * mean_entropy

    def _update_policy(self, dataset):
        """Update the policy using a given dataset.

        The policy is updated via CG and line search.
        """

        assert 'state' in dataset[0]
        assert 'action' in dataset[0]
        assert 'adv' in dataset[0]

        # Use full-batch
        xp = self.policy.xp
        states = batch_states([b['state'] for b in dataset], xp, self.phi)
        if self.obs_normalizer:
            states = self.obs_normalizer(states, update=False)
        actions = xp.array([b['action'] for b in dataset])
        advs = xp.array([b['adv'] for b in dataset], dtype=np.float32)
        if self.standardize_advantages:
            mean_advs = xp.mean(advs)
            std_advs = xp.std(advs)
            advs = (advs - mean_advs) / (std_advs + 1e-8)

        # Recompute action distributions for batch backprop
        action_distrib = self.policy(states)

        log_prob_old = xp.array(
            [transition['log_prob'] for transition in dataset],
            dtype=np.float32)

        gain = self._compute_gain(
            log_prob=action_distrib.log_prob(actions),
            log_prob_old=log_prob_old,
            entropy=action_distrib.entropy,
            advs=advs)

        # Distribution to compute KL div against
        action_distrib_old = action_distrib.copy()

        full_step = self._compute_kl_constrained_step(
            action_distrib=action_distrib,
            action_distrib_old=action_distrib_old,
            gain=gain)

        self._line_search(
            full_step=full_step,
            dataset=dataset,
            advs=advs,
            action_distrib_old=action_distrib_old,
            gain=gain)

    def _update_policy_recurrent(self, dataset):
        """Update the policy using a given dataset.

        The policy is updated via CG and line search.
        """

        xp = self.model.xp
        flat_transitions = list(itertools.chain.from_iterable(dataset))

        # Prepare data for a recurrent model
        seqs_states = []
        for ep in dataset:
            states = self.batch_states(
                [transition['state'] for transition in ep], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            seqs_states.append(states)

        flat_actions = xp.array(
            [transition['action'] for transition in flat_transitions])
        flat_advs = xp.array(
            [transition['adv'] for transition in flat_transitions],
            dtype=np.float32)

        if self.standardize_advantages:
            mean_advs = xp.mean(flat_advs)
            std_advs = xp.std(flat_advs)
            flat_advs = (flat_advs - mean_advs) / (std_advs + 1e-8)

        with chainer.using_config('train', False),\
                chainer.no_backprop_mode():
            policy_rs = self.policy.concatenate_recurrent_states(
                [ep[0]['recurrent_state'][0] for ep in dataset])

        flat_distribs, _ = self.policy.n_step_forward(
            seqs_states, recurrent_state=policy_rs, output_mode='concat')

        log_prob_old = xp.array(
            [transition['log_prob'] for transition in flat_transitions],
            dtype=np.float32)

        gain = self._compute_gain(
            log_prob=flat_distribs.log_prob(flat_actions),
            log_prob_old=log_prob_old,
            entropy=flat_distribs.entropy,
            advs=flat_advs)

        # Distribution to compute KL div against
        action_distrib_old = flat_distribs.copy()

        full_step = self._compute_kl_constrained_step(
            action_distrib=flat_distribs,
            action_distrib_old=action_distrib_old,
            gain=gain)

        self._line_search(
            full_step=full_step,
            dataset=dataset,
            advs=flat_advs,
            action_distrib_old=action_distrib_old,
            gain=gain)

    def _compute_kl_constrained_step(self, action_distrib, action_distrib_old,
                                     gain):
        """Compute a step of policy parameters with a KL constraint."""
        policy_params = _get_ordered_params(self.policy)
        kl = F.mean(action_distrib_old.kl(action_distrib))

        # Check if kl computation fully supports double backprop
        old_style_funcs = _find_old_style_function([kl])
        if old_style_funcs:
            raise RuntimeError("""\
Old-style functions (chainer.Function) are used to compute KL divergence.
Since TRPO requires second-order derivative of KL divergence, its computation
should be done with new-style functions (chainer.FunctionNode) only.

Found old-style functions: {}""".format(old_style_funcs))

        kl_grads = chainer.grad([kl], policy_params,
                                enable_double_backprop=True)
        assert all(g is not None for g in kl_grads), "\
The gradient contains None. The policy may have unused parameters."
        flat_kl_grads = _flatten_and_concat_variables(kl_grads)

        def fisher_vector_product_func(vec):
            fvp = _hessian_vector_product(flat_kl_grads, policy_params, vec)
            return fvp + self.conjugate_gradient_damping * vec

        gain_grads = chainer.grad([gain], policy_params)
        assert all(g is not None for g in kl_grads), "\
The gradient contains None. The policy may have unused parameters."
        flat_gain_grads = _flatten_and_concat_ndarrays(gain_grads)
        step_direction = chainerrl.misc.conjugate_gradient(
            fisher_vector_product_func, flat_gain_grads,
            max_iter=self.conjugate_gradient_max_iter,
        )

        # We want a step size that satisfies KL(old|new) < max_kl.
        # Let d = alpha * step_direction be the actual parameter updates.
        # The second-order approximation of KL divergence is:
        #   KL(old|new) = 1/2 d^T I d + O(||d||^3),
        # where I is a Fisher information matrix.
        # Substitute d = alpha * step_direction and solve KL(old|new) = max_kl
        # for alpha to get the step size that tightly satisfies the constraint.

        dId = float(step_direction.dot(
            fisher_vector_product_func(step_direction)))
        scale = (2.0 * self.max_kl / (dId + 1e-8)) ** 0.5
        return scale * step_direction

    def _line_search(self, full_step, dataset, advs, action_distrib_old, gain):
        """Do line search for a safe step size."""
        xp = self.policy.xp
        policy_params = _get_ordered_params(self.policy)
        policy_params_sizes = [param.size for param in policy_params]
        policy_params_shapes = [param.shape for param in policy_params]
        step_size = 1.0
        flat_params = _flatten_and_concat_ndarrays(policy_params)

        if self.recurrent:
            seqs_states = []
            for ep in dataset:
                states = self.batch_states(
                    [transition['state'] for transition in ep], xp, self.phi)
                if self.obs_normalizer:
                    states = self.obs_normalizer(states, update=False)
                seqs_states.append(states)
            with chainer.using_config('train', False),\
                    chainer.no_backprop_mode():
                policy_rs = self.policy.concatenate_recurrent_states(
                    [ep[0]['recurrent_state'][0] for ep in dataset])

            def evaluate_current_policy():
                distrib, _ = self.policy.n_step_forward(
                    seqs_states, recurrent_state=policy_rs,
                    output_mode='concat')
                return distrib
        else:
            states = self.batch_states(
                [transition['state'] for transition in dataset], xp, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)

            def evaluate_current_policy():
                return self.policy(states)

        flat_transitions = (list(itertools.chain.from_iterable(dataset))
                            if self.recurrent else dataset)
        actions = xp.array(
            [transition['action'] for transition in flat_transitions])
        log_prob_old = xp.array(
            [transition['log_prob'] for transition in flat_transitions],
            dtype=np.float32)

        for i in range(self.line_search_max_backtrack + 1):
            self.logger.info(
                'Line search iteration: %s step size: %s', i, step_size)
            new_flat_params = flat_params + step_size * full_step
            new_params = _split_and_reshape_to_ndarrays(
                new_flat_params,
                sizes=policy_params_sizes,
                shapes=policy_params_shapes,
            )
            _replace_params_data(policy_params, new_params)
            with chainer.using_config('train', False),\
                    chainer.no_backprop_mode():
                new_action_distrib = evaluate_current_policy()
                new_gain = self._compute_gain(
                    log_prob=new_action_distrib.log_prob(actions),
                    log_prob_old=log_prob_old,
                    entropy=new_action_distrib.entropy,
                    advs=advs)
                new_kl = F.mean(action_distrib_old.kl(new_action_distrib))

            improve = new_gain.array - gain.array
            self.logger.info(
                'Surrogate objective improve: %s', float(improve))
            self.logger.info('KL divergence: %s', float(new_kl.array))
            if not xp.isfinite(new_gain.array):
                self.logger.info(
                    "Surrogate objective is not finite. Bakctracking...")
            elif not xp.isfinite(new_kl.array):
                self.logger.info(
                    "KL divergence is not finite. Bakctracking...")
            elif improve < 0:
                self.logger.info(
                    "Surrogate objective didn't improve. Bakctracking...")
            elif float(new_kl.array) > self.max_kl:
                self.logger.info(
                    "KL divergence exceeds max_kl. Bakctracking...")
            else:
                self.kl_record.append(float(new_kl.array))
                self.policy_step_size_record.append(step_size)
                break
            step_size *= 0.5
        else:
            self.logger.info("\
Line search coundn't find a good step size. The policy was not updated.")
            self.policy_step_size_record.append(0.)
            _replace_params_data(
                policy_params,
                _split_and_reshape_to_ndarrays(
                    flat_params,
                    sizes=policy_params_sizes,
                    shapes=policy_params_shapes),
            )

    def act_and_train(self, obs, reward):

        if self.last_state is not None:
            transition = {
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': obs,
                'nonterminal': 1.0,
            }
            if self.recurrent:
                transition['recurrent_state'] =\
                    self.model.get_recurrent_state_at(
                        self.train_prev_recurrent_states,
                        0, unwrap_variable=True)
                self.train_prev_recurrent_states = None
                transition['next_recurrent_state'] =\
                    self.model.get_recurrent_state_at(
                        self.train_recurrent_states, 0, unwrap_variable=True)
            self.last_episode.append(transition)

        self._update_if_dataset_is_ready()

        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        # action_distrib will be recomputed when computing gradients
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (action_distrib, value), self.train_recurrent_states =\
                    self.model(b_state, self.train_prev_recurrent_states)
            else:
                action_distrib, value = self.model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().array)[0]
            self.entropy_record.append(float(action_distrib.entropy.array))
            self.value_record.append(float(value.array))

        self.last_state = obs
        self.last_action = action

        return action

    def act(self, obs):
        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                action_distrib, self.test_recurrent_states =\
                    self.policy(b_state, self.test_recurrent_states)
            else:
                action_distrib = self.policy(b_state)
            if self.act_deterministically:
                action = chainer.cuda.to_cpu(
                    action_distrib.most_probable.array)[0]
            else:
                action = chainer.cuda.to_cpu(
                    action_distrib.sample().array)[0]

        return action

    def stop_episode_and_train(self, state, reward, done=False):

        assert self.last_state is not None
        transition = {
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'next_state': state,
            'nonterminal': 0.0 if done else 1.0,
        }
        if self.recurrent:
            transition['recurrent_state'] = self.model.get_recurrent_state_at(
                self.train_prev_recurrent_states, 0, unwrap_variable=True)
            self.train_prev_recurrent_states = None
            transition['next_recurrent_state'] =\
                self.model.get_recurrent_state_at(
                    self.train_recurrent_states, 0, unwrap_variable=True)
            self.train_recurrent_states = None
        self.last_episode.append(transition)

        self.last_state = None
        self.last_action = None

        self._flush_last_episode()
        self.stop_episode()

        self._update_if_dataset_is_ready()

    def stop_episode(self):
        self.test_recurrent_states = None

    def batch_act(self, batch_obs):
        xp = self.xp
        b_state = self.batch_states(batch_obs, xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                (action_distrib, _), self.test_recurrent_states = self.model(
                    b_state, self.test_recurrent_states)
            else:
                action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                action = chainer.cuda.to_cpu(
                    action_distrib.most_probable.array)
            else:
                action = chainer.cuda.to_cpu(action_distrib.sample().array)

        return action

    def batch_act_and_train(self, batch_obs):
        xp = self.xp
        b_state = self.batch_states(batch_obs, xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs

        # action_distrib will be recomputed when computing gradients
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (action_distrib, batch_value), self.train_recurrent_states =\
                    self.model(b_state, self.train_prev_recurrent_states)
            else:
                action_distrib, batch_value = self.model(b_state)
            batch_action = chainer.cuda.to_cpu(action_distrib.sample().array)
            self.entropy_record.extend(
                chainer.cuda.to_cpu(action_distrib.entropy.array))
            self.value_record.extend(chainer.cuda.to_cpu((batch_value.array)))

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i for i, (done, reset)
                in enumerate(zip(batch_done, batch_reset)) if done or reset]
            if indices_that_ended:
                self.test_recurrent_states =\
                    self.model.mask_recurrent_state_at(
                        self.test_recurrent_states, indices_that_ended)

    def batch_observe_and_train(self, batch_obs, batch_reward,
                                batch_done, batch_reset):

        for i, (state, action, reward, next_state, done, reset) in enumerate(zip(  # NOQA
            self.batch_last_state,
            self.batch_last_action,
            batch_reward,
            batch_obs,
            batch_done,
            batch_reset,
        )):
            if state is not None:
                assert action is not None
                transition = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'nonterminal': 0.0 if done else 1.0,
                }
                if self.recurrent:
                    transition['recurrent_state'] =\
                        self.model.get_recurrent_state_at(
                            self.train_prev_recurrent_states,
                            i, unwrap_variable=True)
                    transition['next_recurrent_state'] =\
                        self.model.get_recurrent_state_at(
                            self.train_recurrent_states,
                            i, unwrap_variable=True)
                self.batch_last_episode[i].append(transition)
            if done or reset:
                assert self.batch_last_episode[i]
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
            self.batch_last_state[i] = None
            self.batch_last_action[i] = None

        self.train_prev_recurrent_states = None

        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i for i, (done, reset)
                in enumerate(zip(batch_done, batch_reset)) if done or reset]
            if indices_that_ended:
                self.train_recurrent_states =\
                    self.model.mask_recurrent_state_at(
                        self.train_recurrent_states, indices_that_ended)

        self._update_if_dataset_is_ready()

    def get_statistics(self):
        return [
            ('average_value', _mean_or_nan(self.value_record)),
            ('average_entropy', _mean_or_nan(self.entropy_record)),
            ('average_kl', _mean_or_nan(self.kl_record)),
            ('average_policy_step_size',
                _mean_or_nan(self.policy_step_size_record)),
            ('explained_variance', self.explained_variance),
        ]
