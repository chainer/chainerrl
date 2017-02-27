import random

import chainer
import numpy as np

import chainerrl


def opponent_of(role):
    return {'O': 'X', 'X': 'O'}[role]


def is_win(role, board):
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
             (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    return any(all(board[i] == role for i in line) for line in lines)


class TicTacToeEnv(object):

    def __init__(self):
        self.roles = ['X', 'O']

    def reset(self):
        self.board = [None] * 9
        self.active_role = 'X'
        return (self.active_role, self.board)

    def get_legal_actions(self):
        return [i for i in range(9) if self.board[i] is None]

    def step(self, action):
        if self.board[action]:
            # Illegal actions are replaced by legal random actions
            action = random.choice(self.get_legal_actions())

        self.board[action] = self.active_role

        if is_win(self.active_role, self.board):
            # Win
            rewards = {self.active_role: 1,
                       opponent_of(self.active_role): -1}
            return None, rewards, True, {}

        if not self.get_legal_actions():
            # Draw
            rewards = {player: 0 for player in self.roles}
            return None, rewards, True, {}

        self.active_role = opponent_of(self.active_role)

        rewards = {player: 0 for player in self.roles}
        return (self.active_role, self.board), rewards, False, {}


def feature_extractor(obs):
    if obs is None:
        return np.zeros(9 * 3, dtype=np.float32)
    role, board = obs
    assert len(board) == 9
    my_marks = [x == role for x in board]
    opponent = opponent_of(role)
    opponent_marks = [x == opponent for x in board]
    empty = [x is None for x in board]
    return np.asarray(my_marks + opponent_marks + empty, dtype=np.float32)


def play_against_random_player(role, agent):
    env = TicTacToeEnv()
    obs = env.reset()
    done = False
    while not done:
        active_role = obs[0]
        if active_role == role:
            a = agent.act(obs)
        else:
            a = random.choice(env.get_legal_actions())
        obs, r, done, _ = env.step(a)
    agent.stop_episode()
    return r[role]


def eval_against_random_player(agents):
    n_runs = 100
    for role, agent in agents.items():
        Rs = []
        for _ in range(n_runs):
            R = play_against_random_player(role, agent)
            Rs.append(R)
        average_R = sum(Rs) / n_runs
        print(role, average_R)


def main():
    env = TicTacToeEnv()
    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        9 * 3, 9, n_hidden_channels=100, n_hidden_layers=2)
    q_func.to_gpu(0)
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        0.3, lambda: np.random.randint(9))
    opt = chainer.optimizers.Adam(eps=1e-2)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(10 ** 5)
    opt.setup(q_func)
    agents = {}
    for role in env.roles:
        agents[agent_name] = chainerrl.agents.DQN(
            q_func, opt,
            replay_buffer=replay_buffer,
            explorer=explorer,
            gamma=0.99,
            phi=feature_extractor,
            replay_start_size=100,
            target_update_frequency=100,
            update_frequency=1)
        agents[role] = agent

    # Training from self-play
    X_Rs = []
    O_Rs = []
    for nth_episode in range(1000000):
        obs = env.reset()
        r = {role: 0 for role in env.roles}
        done = False
        while not done:
            active_role = obs[0]
            assert r[active_role] == 0
            a = agents[active_role].act_and_train(obs, r[active_role])
            obs, r, done, _ = env.step(a)

        # Every agent must observe the final results
        for role, agent in agents.items():
            # print(role, r[role])
            agent.stop_episode_and_train(obs, r[role], done)

        assert agents['X'] is not agents['O']
        assert agents['X'].model is agents['O'].model
        assert agents['X'].target_model is agents['O'].target_model
        assert agents['X'].optimizer is agents['O'].optimizer

        X_Rs.append(r['X'])
        O_Rs.append(r['O'])

        if nth_episode % 10000 == 0:
            print(nth_episode)
            X_Rs = []
            O_Rs = []
            for role, agent in agents.items():
                print(role, agent.get_statistics())
            eval_against_random_player(agents)
            agents['X'].save('tictactoe_{}'.format(nth_episode))


if __name__ == '__main__':
    main()
