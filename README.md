# ChainerRL

ChainerRL is a deep reinforcement learning library built on top of Chainer.

## Requirements

For Python 3.5.1+, requirements are:

- atari_py
- chainer>=1.20.1
- cached-property
- future
- gym
- numpy>=1.10.4
- pillow
- scipy

For Python 2.7.6+, you need additional requirements. See requirements.txt.

## Agents

| Agent | Discrete Action | Continous Action | Recurrent Model | CPU Async Training |
|:------|:---------------:|:----------------:|:---------------:|:--------------:|
| DQN (including DoubleDQN etc.) | o | o (NAF) | o | x |
| DDPG | x | o | o | x |
| A3C | o | o | o | o |
| NSQ (N-step Q-learning) | o | o (NAF) | o | o |

Following agents have been implemented: 
- A3C (Asynchronous Advantage Actor-Critic)
- Asynchronous N-step Q-learning
- DQN (including Double DQN, Persistent Advantage Learning (PAL), Double PAL, Dynamic Policy Programming (DPP))
- DDPG (Deep Deterministic Poilcy Gradients)
- PGT (Policy Gradient Theorem)

Q-function based agents can utilize Normalized Advantage Functions (NAFs) to tackle continuous-action problems as well as DQN-like discrete output networks.

## Environments

Environments that support OpenAI Gym's interface (`reset`, `step` and `close` functions) can be used.

Additionally, following environments have been implemented in this library:
- ALE (https://github.com/mgbellemare/Arcade-Learning-Environment)

## How to use

To get started,
```
pip install -r requirements.txt
python setup.py develop
```
Please see the examples in the `examples` directory.

## How to test

To test chainerrl modules, install `nose` and run `nosetests`.

To test examples, run `test_examples.sh`.
