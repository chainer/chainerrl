# ChainerRL
[![Build Status](https://travis-ci.org/pfnet/chainerrl.svg?branch=master)](https://travis-ci.org/pfnet/chainerrl)

ChainerRL is a deep reinforcement learning library that implements various state-of-the-art deep reinforcement algorithms in Python using [Chainer](https://github.com/pfnet/chainer), a flexible deep learning framework.

## Installation

ChainerRL is tested with Python 2.7+ and 3.5.1+. For other requirements, see [requirements.txt](requirements.txt).

ChainerRL can be installed via PyPI:
```
pip install chainerrl
```

It can also be installed from the source code:
```
python setup.py install
```

* Windows user

ChainerRL contains `atari_py` as dependencies, and windows users may face error while installing it. 
This problem is discussed in [OpenAI gym issues](https://github.com/openai/gym/issues/11), 
and one possible counter measure is to enable "Bash on Ubuntu on Windows" for Windows 10 users.

Refer [Official install guilde](https://msdn.microsoft.com/en-us/commandline/wsl/install_guide) to install "Bash on Ubuntu on Windows". 

## Getting started

You can try [ChainerRL Quickstart Guide](examples/quickstart/quickstart.ipynb) first, or check the [examples](examples) ready for Atari 2600 and Open AI Gym.

## Algorithms

| Algorithm | Discrete Action | Continous Action | Recurrent Model | CPU Async Training |
|:----------|:---------------:|:----------------:|:---------------:|:------------------:|
| DQN (including DoubleDQN etc.) | o | o (NAF) | o | x |
| DDPG | x | o | o | x |
| A3C | o | o | o | o |
| ACER | o | o | o | o |
| NSQ (N-step Q-learning) | o | o (NAF) | o | o |

Following algorithms have been implemented in ChainerRL:
- A3C (Asynchronous Advantage Actor-Critic)
- ACER (Actor-Critic with Experience Replay)
- Asynchronous N-step Q-learning
- DQN (including Double DQN, Persistent Advantage Learning (PAL), Double PAL, Dynamic Policy Programming (DPP))
- DDPG (Deep Deterministic Poilcy Gradients) (including SVG(0))
- PGT (Policy Gradient Theorem)

Q-function based algorithms such as DQN can utilize a Normalized Advantage Function (NAF) to tackle continuous-action problems as well as DQN-like discrete output networks.

## Environments

Environments that support the subset of OpenAI Gym's interface (`reset` and `step` methods) can be used.

## Contributing

Any kind of contribution to ChainerRL would be highly appreciated! If you are interested in contributing to ChainerRL, please read [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT License](LICENSE).
