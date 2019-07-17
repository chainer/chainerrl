<div align="center"><img src="https://raw.githubusercontent.com/chainer/chainerrl/master/assets/ChainerRL.png" width="400"/></div>

# ChainerRL
[![Build Status](https://travis-ci.org/chainer/chainerrl.svg?branch=master)](https://travis-ci.org/chainer/chainerrl)
[![Coverage Status](https://coveralls.io/repos/github/chainer/chainerrl/badge.svg?branch=master)](https://coveralls.io/github/chainer/chainerrl?branch=master)
[![Documentation Status](https://readthedocs.org/projects/chainerrl/badge/?version=latest)](http://chainerrl.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/chainerrl.svg)](https://pypi.python.org/pypi/chainerrl)

ChainerRL is a deep reinforcement learning library that implements various state-of-the-art deep reinforcement algorithms in Python using [Chainer](https://github.com/chainer/chainer), a flexible deep learning framework.

![Breakout](assets/breakout.gif)
![Humanoid](assets/humanoid.gif)
![Grasping](assets/grasping.gif)

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
| Rainbow | ✓ | x | ✓ | x |
| IQN | ✓ | x | x | x |
| DDPG | x | ✓ | ✓ | x |
| A3C  | ✓ | ✓ | ✓ | ✓ |
| ACER | ✓ | ✓ | ✓ | ✓ |
| NSQ (N-step Q-learning) | ✓ | ✓ (NAF) | ✓ | ✓ |
| PCL (Path Consistency Learning) | ✓ | ✓ | ✓ | ✓ |
| PPO  | ✓ | ✓ | ✓ | x |
| TRPO | ✓ | ✓ | x | x |
| TD3 | x | ✓ | x | x |

Following algorithms have been implemented in ChainerRL:
- [A3C (Asynchronous Advantage Actor-Critic)](https://arxiv.org/abs/1602.01783)
  - examples: [[atari reproduction]](examples/atari/reproduction/a3c) [[atari]](examples/atari/train_a3c_ale.py) [[general gym]](examples/gym/train_a3c_gym.py)
- [ACER (Actor-Critic with Experience Replay)](https://arxiv.org/abs/1611.01224)
  - examples: [[atari]](examples/atari/train_acer_ale.py) [[general gym]](examples/gym/train_acer_gym.py)
- [Asynchronous N-step Q-learning](https://arxiv.org/abs/1602.01783)
  - examples: [[atari]](examples/atari/train_nsq_ale.py) [[general gym]](examples/gym/train_nsq_gym.py)
- [Rainbow](https://arxiv.org/abs/1710.02298)
  - examples: [[atari reproduction]](examples/atari/reproduction/rainbow)
- [Categorical DQN](https://arxiv.org/abs/1707.06887)
  - examples: [[atari]](examples/atari/train_categorical_dqn_ale.py) [[general gym]](examples/gym/train_categorical_dqn_gym.py)
- [IQN (Implicit Quantile Networks)](https://arxiv.org/abs/1806.06923)
  - examples: [[atari reproduction]](examples/atari/reproduction/iqn) [[atari]](examples/atari/train_iqn_ale.py) [[general gym]](examples/gym/train_iqn_gym.py)
- [DQN (Deep Q-Netowrk)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (including [Double DQN](https://arxiv.org/abs/1509.06461), [Persistent Advantage Learning (PAL)](https://arxiv.org/abs/1512.04860), Double PAL, [Dynamic Policy Programming (DPP)](http://www.jmlr.org/papers/volume13/azar12a/azar12a.pdf))
  - examples: [[atari reproduction]](examples/atari/reproduction/dqn) [[atari]](examples/atari/train_dqn_ale.py) [[atari (batched)]](examples/atari/train_dqn_batch_ale.py) [[general gym]](examples/gym/train_dqn_gym.py)
- [DDPG (Deep Deterministic Policy Gradients)](https://arxiv.org/abs/1509.02971) (including [SVG(0)](https://arxiv.org/abs/1510.09142))
  - examples: [[mujoco reproduction]](examples/mujoco/reproduction/ddpg) [[mujoco]](examples/mujoco/train_ddpg_gym.py) [[mujoco (batched)]](examples/mujoco/train_ddpg_batch_gym.py)
- [PCL (Path Consistency Learning)](https://arxiv.org/abs/1702.08892)
  - examples: [[general gym]](examples/gym/train_pcl_gym.py)
- [PPO (Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347)
  - examples: [[mujoco reproduction]](examples/mujoco/reproduction/ppo) [[atari]](examples/atari/train_ppo_ale.py) [[mujoco]](examples/mujoco/train_ppo_gym.py) [[mujoco (batched)]](examples/mujoco/train_ppo_batch_gym.py)
- [TRPO (Trust Region Policy Optimization)](https://arxiv.org/abs/1502.05477) with [GAE (Generalized Advantage Estimation)](https://arxiv.org/abs/1506.02438)
  - examples: [[mujoco]](examples/mujoco/train_trpo_gym.py)
- [TD3 (Twin Delayed Deep Deterministic policy gradient algorithm)](https://arxiv.org/abs/1802.09477)
  - examples: [[mujoco reproduction]](examples/mujoco/reproduction/td3)

Q-function based algorithms such as DQN can utilize a Normalized Advantage Function (NAF) to tackle continuous-action problems as well as DQN-like discrete output networks.

## Paper Implementations
The following papers have been implemented in ChainerRL:
- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Human-level control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
- [Increasing the Action Gap: New Operators for Reinforcement Learning](https://arxiv.org/abs/1512.04860)
- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)
- [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)
- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)


## Visualization

ChainerRL has a set of accompanying [visualization tools](https://github.com/chainer/chainerrl-visualizer) in order to aid developers' ability to understand and debug their RL agents. With this visualization tool, the behavior of ChainerRL agents can be easily inspected from a browser UI.


## Environments

Environments that support the subset of OpenAI Gym's interface (`reset` and `step` methods) can be used.

## Contributing

Any kind of contribution to ChainerRL would be highly appreciated! If you are interested in contributing to ChainerRL, please read [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT License](LICENSE).
