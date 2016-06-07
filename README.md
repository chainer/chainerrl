# Drill

Drill is a deep reinforcement learning library, still work-in-progress.

## Requirements

- Python 3.5.1+
- Chainer 1.8.2+

## Agents

Following agents have been implemented: 
- A3C
- Asynchronous N-step Q-learning (work-in-progress)
- DQN (including Double DQN, Persistent Advantage Learning (PAL), Double PAL)

## Environments

Environments that support OpenAI Gym's interface can be used.

Additionally, following environments have been implemented in this library:
- ALE (https://github.com/mgbellemare/Arcade-Learning-Environment)
- VizDoom

## How to use

Please see the examples: `train_a3c_*.py` for A3C, `train_dqn_*.py` for DQN.
