import random

import chainer
import numpy as np

import chainerrl


def opponent_of(player):
    return {'O': 'X', 'X': 'O'}[player]


def is_win(player, board):
    return (any(all(board[col + row * 3] == player for col in range(3))
                for row in range(3)) or
            any(all(board[col + row * 3] == player for row in range(3))
                for col in range(3)) or
            all(board[i + i * 3] == player for i in range(3)) or
            all(board[i + (2 - i) * 3] == player for i in range(3)))


class TicTacToeEnv(object):

    def __init__(self):
        self.agents = ['X', 'O']

    def reset(self):
        self.board = [None] * 9
        self.current_player = 'X'
        return (self.current_player, self.board)

    def get_legal_actions(self):
        return [i for i in range(9) if self.board[i] is None]

    def step(self, action):
        if self.board[action]:
            # Illegal actions are replaced by legal random actions
            action = random.choice(self.get_legal_actions())

        self.board[action] = self.current_player

        if is_win('X', self.board) or is_win('O', self.board):
            # Win
            rewards = {self.current_player: 1,
                       opponent_of(self.current_player): -1}
            return None, rewards, True, {}

        if all(x is not None for x in self.board):
            # Draw
            rewards = {self.current_player: 0,
                       opponent_of(self.current_player): 0}
            return None, rewards, True, {}

        self.current_player = opponent_of(self.current_player)

        rewards = {self.current_player: 0,
                   opponent_of(self.current_player): 0}
        return (self.current_player, self.board), rewards, False, {}


def feature_extractor(obs):
    if obs is None:
        return np.zeros(9 * 3, dtype=np.float32)
    player, board = obs
    assert len(board) == 9
    my_marks = [x == player for x in board]
    opponent = opponent_of(player)
    opponent_marks = [x == opponent for x in board]
    empty = [x is None for x in board]
    return np.asarray(my_marks + opponent_marks + empty, dtype=np.float32)


def play_against_random_player(agent_name, agent):
    env = TicTacToeEnv()
    obs = env.reset()
    done = False
    while not done:
        current_player = obs[0]
        if current_player == agent_name:
            a = agent.act(obs)
        else:
            a = random.choice(env.get_legal_actions())
        obs, r, done, _ = env.step(a)
    agent.stop_episode()
    return r[agent_name]


def eval_against_random_player(agents):
    for agent_name, agent in agents.items():
        Rs = []
        for _ in range(100):
            R = play_against_random_player(agent_name, agent)
            Rs.append(R)
        average_R = sum(Rs) / 100
        print(agent_name, average_R)


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
    for agent_name in env.agents:
        agents[agent_name] = chainerrl.agents.DQN(
            q_func, opt,
            replay_buffer=replay_buffer,
            explorer=explorer,
            gamma=0.99,
            phi=feature_extractor,
            replay_start_size=100,
            target_update_frequency=100,
            update_frequency=1)

    # Training from self-play
    for nth_episode in range(10000):
        obs = env.reset()
        r = {agent_name: 0 for agent_name in env.agents}
        done = False
        while not done:
            current_player = obs[0]
            a = agents[current_player].act_and_train(obs, r[current_player])
            obs, r, done, _ = env.step(a)

        # Every agent must observe the final results
        for agent_name, agent in agents.items():
            agent.stop_episode_and_train(obs, r[agent_name], done)

        if nth_episode % 1000 == 0:
            for agent_name, agent in agents.items():
                print(agent_name, agent.get_statistics())
            eval_against_random_player(agents)


if __name__ == '__main__':
    main()
