<div align="center"><img src="assets/ChainerRL.png" width="400"/></div>

# ChainerRL
[![Build Status](https://travis-ci.org/chainer/chainerrl.svg?branch=master)](https://travis-ci.org/chainer/chainerrl)
[![Coverage Status](https://coveralls.io/repos/github/chainer/chainerrl/badge.svg?branch=master)](https://coveralls.io/github/chainer/chainerrl?branch=master)
[![Documentation Status](https://readthedocs.org/projects/chainerrl/badge/?version=latest)](http://chainerrl.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/chainerrl.svg)](https://pypi.python.org/pypi/chainerrl)

ChainerRL is a deep reinforcement learning library that implements various state-of-the-art deep reinforcement algorithms in Python using [Chainer](https://github.com/pfnet/chainer), a flexible deep learning framework.

![Breakout](assets/breakout.gif)
![Humanoid](assets/humanoid.gif)

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

Refer to [Installation](http://chainerrl.readthedocs.io/en/latest/install.html) for more information on installation. 

## Getting started

You can try [ChainerRL Quickstart Guide](examples/quickstart/quickstart.ipynb) first, or check the [examples](examples) ready for Atari 2600 and Open AI Gym.

For more information, you can refer to [ChainerRL's documentation](http://chainerrl.readthedocs.io/en/latest/index.html).

## Algorithms

| Algorithm | Discrete Action | Continous Action | Recurrent Model | CPU Async Training |
|:----------|:---------------:|:----------------:|:---------------:|:------------------:|
| DQN (including DoubleDQN etc.) | ✓ | ✓ (NAF) | ✓ | x |
| Categorical DQN | ✓ | x | ✓ | x |
| DDPG | x | ✓ | ✓ | x |
| A3C  | ✓ | ✓ | ✓ | ✓ |
| ACER | ✓ | ✓ | ✓ | ✓ |
| NSQ (N-step Q-learning) | ✓ | ✓ (NAF) | ✓ | ✓ |
| PCL (Path Consistency Learning) | ✓ | ✓ | ✓ | ✓ |
| PPO  | ✓ | ✓ | x | x |
| TRPO | ✓ | ✓ | x | x |

Following algorithms have been implemented in ChainerRL:
- A3C (Asynchronous Advantage Actor-Critic)
- ACER (Actor-Critic with Experience Replay)
- Asynchronous N-step Q-learning
- Categorical DQN
- DQN (including Double DQN, Persistent Advantage Learning (PAL), Double PAL, Dynamic Policy Programming (DPP))
- DDPG (Deep Deterministic Policy Gradients) (including SVG(0))
- PGT (Policy Gradient Theorem)
- PCL (Path Consistency Learning)
- PPO (Proximal Policy Optimization)
- TRPO (Trust Region Policy Optimization)

Q-function based algorithms such as DQN can utilize a Normalized Advantage Function (NAF) to tackle continuous-action problems as well as DQN-like discrete output networks.

## Paper Implementations
The following papers have been implemented in ChainerRL:
- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Human-level control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf)
- [Increasing the Action Gap: New Operators for Reinforcement Learning](https://arxiv.org/pdf/1512.04860.pdf)
- [Noisy Networks for Exploration](https://arxiv.org/pdf/1706.10295.pdf)
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
- [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)
- [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/pdf/1611.01224.pdf)
- [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/pdf/1702.08892.pdf)

## Environments

Environments that support the subset of OpenAI Gym's interface (`reset` and `step` methods) can be used.

## Contributing

Any kind of contribution to ChainerRL would be highly appreciated! If you are interested in contributing to ChainerRL, please read [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT License](LICENSE).
