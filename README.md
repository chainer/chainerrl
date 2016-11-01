# Chainer-RL

Chainer-RL is a deep reinforcement learning library built on top of Chainer.

## Requirements

- Python 2.7.6+, 3.4.3+, 3.5.1+
- Chainer 1.16.0+
- cached-property
- fastcache
- h5py
- statistics
- future

## Get started



## Agents

Following agents have been implemented: 
- A3C (Asynchronous Advantage Actor-Critic)
- Asynchronous N-step Q-learning (work-in-progress)
- DQN (including Double DQN, Persistent Advantage Learning (PAL), Double PAL, Dynamic Policy Programming (DPP))
- DDPG (Deep Deterministic Poilcy Gradients)
- PGT (Policy Gradient Theorem)

Q-function based agents can utilize Normalized Advantage Functions (NAFs) to tackle continuous-action problems as well as DQN-like discrete output networks.

## Environments

Environments that support OpenAI Gym's interface (`reset`, `step` and `close` functions) can be used.

Additionally, following environments have been implemented in this library:
- ALE (https://github.com/mgbellemare/Arcade-Learning-Environment)
- VizDoom

## How to use

To get started,
```
pip install -r requirements.txt
python setup.py develop
```
Please see the examples in the `examples` directory.
