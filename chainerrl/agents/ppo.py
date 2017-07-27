import chainer
import chainer.functions as F
import copy


from chainerrl import agent


def _F_clip(x, x_min, x_max):
    return F.minimum(F.maximum(x, x_min), x_max)


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
                 clip_eps_vf=0.2,
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
        self.clip_eps_vf = clip_eps_vf

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
                + (self.gamma * transition['nonterminal']
                   * transition['next_v_pred'])
                - transition['v_pred']
                )
            adv = td_err + self.lambd * adv
            transition['adv'] = adv
            transition['v_teacher'] = adv + transition['v_pred']  # ????

    def lossfun(self, prob_ratio, advs, vs_pred, vs_pred_old, vs_teacher, ent):
        prob_ratio = F.expand_dims(prob_ratio, axis=-1)
        loss_policy = - F.mean(F.minimum(
            prob_ratio * advs,
            F.clip(prob_ratio, 1-self.clip_eps, 1+self.clip_eps) * advs))

        # loss_value_func = F.mean_squared_error(vs_pred, vs_teacher)
        loss_value_func = F.mean(F.maximum(
            F.square(vs_pred - vs_teacher),
            F.square(_F_clip(vs_pred,
                             vs_pred_old - self.clip_eps_vf,
                             vs_pred_old + self.clip_eps_vf)
                     - vs_teacher)
            ))

        loss_entropy = F.mean(ent)

        # print((loss_policy, loss_value_func, loss_entropy))
        return (
            loss_policy
            + self.value_func_coeff * loss_value_func
            + self.entropy_coeff * loss_entropy
            )

    def update(self):
        xp = self.xp
        target_model = copy.deepcopy(self.model)
        dataset_iter = chainer.iterators.SerialIterator(
            self.memory, self.batchsize)

        dataset_iter.reset()
        while dataset_iter.epoch < self.epochs:
            batch = dataset_iter.__next__()
            states = xp.array([b['state'] for b in batch], dtype=xp.float32)
            actions = xp.array([b['action'] for b in batch], dtype=xp.int32)
            vs_pred_old = xp.array(
                [b['v_pred'] for b in batch], dtype=xp.float32)

            distribs, vs_pred = self.model(states)
            # print(distribs)
            log_probs = distribs.log_prob(actions)
            target_distribs, _ = target_model(states)
            target_log_probs = target_distribs.log_prob(actions)
            prob_ratio = F.exp(log_probs - target_log_probs)

            advs = xp.array([b['adv'] for b in batch], dtype=xp.float32)
            vs_teacher = xp.array(
                [b['v_teacher'] for b in batch], dtype=xp.float32)
            ent = distribs.entropy

            self.optimizer.update(
                self.lossfun,
                prob_ratio, advs, vs_pred, vs_pred_old, vs_teacher, ent)

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
        action, _ = self._act(state, train=False)
        return action

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
