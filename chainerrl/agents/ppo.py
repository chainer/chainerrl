import chainer
import chainer.functions as F
import copy


from chainerrl import agent
from chainerrl.misc.copy_param import synchronize_parameters


class PPO(agent.AttributeSavingMixin, agent.Agent):
    saved_attributes = ['model', 'optimizer']

    def __init__(self, model, optimizer,
                 gamma=0.99,
                 lambd=0.95,
                 value_func_coeff=1.0,
                 entropy_coeff=0.01,
                 horizon=2048,
                 batchsize=64,
                 epochs=10,
                 clip_eps=0.2,
#                 sync_config={
#                     'method': 'soft',
#                     'tau': 0.99},
                ):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.lambd = lambd
        self.value_func_coeff = value_func_coeff
        self.entropy_coeff = entropy_coeff
        self.horizon = horizon
        self.batchsize = batchsize
        self.epochs = epochs
        self.clip_eps = clip_eps
#        self.sync_config = sync_config

        self.xp = self.model.xp
        self.target_model = None
        self.last_state = None

        self.memory = []
        self.last_episode = []

    def _act(self, state, train):
        state = state.astype(self.xp.float32)
        with chainer.using_config('train', train):
            b_state = F.expand_dims(state, axis=0)
            action_distrib, v = self.model(b_state)
            action = action_distrib.sample()
            return action[0].data, v[0].data

    def _train(self):
        if len(self.memory) + len(self.last_episode) >= self.horizon:
            self.flush_last_episode()
            self.update()
            self.memory = []

#    def sync_target_network(self):
#        if self.target_model is None:
#            self.target_model = copy.deepcopy(self.model)
#        else:
#            synchronize_parameters(
#                src=self.model,
#                dst=self.target_model,
#                **self.sync_config)

    def flush_last_episode(self):
        if self.last_episode:
            self.compute_v_teacher()
            self.memory.extend(self.last_episode)
            self.last_episode = []

    def compute_v_teacher(self):
        adv = 0.0
        for transition in reversed(self.last_episode):
            td_err = (
                transition['reward']
                + self.gamma * transition['nonterminal'] * transition['next_v_pred']
                - transition['v_pred']
                )
            adv = td_err + self.lambd * adv
            transition['adv'] = adv
            transition['v_teacher'] = adv + transition['v_pred']  # ????

    def update(self):
        xp = self.xp
        target_model = copy.deepcopy(self.model)
        dataset = {k: xp.array([e[k] for e in self.memory], dtype=xp.float32)
                   for k in self.memory[0].keys()}
        n = len(dataset)
        ix_iter = chainer.iterators.SerialIterator(range(n), self.batchsize)

        def lossfun(states, actions, rewards, advs, vs_teacher):
            distribs, vs_pred = self.model(states)
            log_probs = distribs.log_prob(actions)
            target_distribs = target_model(states)
            target_log_probs = target_distribs.log_prob(actions)
            prob_ratio = F.exp(log_prob - target_distribs)

            loss_policy = - F.minimum(
                prob_ratio * advs,
                F.clip(prob_ratio, 1-self.clip_eps, 1+self.clip_eps) * advs)

            loss_value_func = F.mean_squared_loss(vs_pred, vs_teacher)

            ent = distribs.entropy()
            loss_entropy = F.mean(ent)

            return (
                loss_policy
                + self.value_func_coeff * loss_value_func
                + self.entropy_coeff * loss_entropy
                )

        ix_iter.reset()
        while ix_iter.epoch < self.epochs:
            ix = ix_iter.__next__()
            self.optimizer.update(
                lossfun,
                dataset['state'][ix],
                dataset['action'][ix],
                dataset['reward'][ix],
                dataset['adv'][ix],
                dataset['v_teacher'][ix])

    def act_and_train(self, state, reward, done=False):
        action, v = self._act(state, train=True)

        if self.last_state is not None:
            self.last_episode.append({
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'v_pred': self.last_v,
                'next_state': state,
                'next_v_pred': v,
                'nonterminal': 0.0 if done else 1.0})
        self.last_state = state
        self.last_action = action
        self.last_v = v

        self._train()
        return action

    def act(self, state):
        return self._act(state, train=False)

    def stop_episode_and_train(self, state, reward, done=False):
        _, v = self._act(state, train=True)

        if self.last_state is not None:
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

        self.flush_last_episode()
        self.stop_episode()

    def stop_episode(self):
        pass

    def get_statistics(self):
        return []
