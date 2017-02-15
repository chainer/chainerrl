# ChainerRL

ChainerRL is a deep reinforcement learning library that implements various state-of-the-art deep reinforcement algorithms in Python using Chainer, a flexible deep learning framework.

## Installation

ChainerRL can be installed via PyPI:
```
pip install chainerrl
```

It can also be installed from the source code:
```
python setup.py install
```

## Getting started

You can try [ChainerRL Quickstart Guide](examples/quickstart/quickstart.ipynb) first, or check the [examples](examples) ready for Atari 2600 and Open AI Gym.

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

Q-function based agents can utilize a Normalized Advantage Function (NAF) to tackle continuous-action problems as well as DQN-like discrete output networks.

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
