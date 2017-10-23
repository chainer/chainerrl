# Examples for Arcade Learning Environment

- `train_a3c_ale.py`: A3C
- `train_acer_ale.py`: DiscreteACER
- `train_nsq_ale.py`: NSQ (n-step Q-learning)
- `train_dqn_ale.py`: DQN, DoubleDQN or PAL

## Requirements

- atari_py>=0.1.1
- OpenCV (optional)

## How to run

```
python train_a3c_ale.py n_processes rom [options]
python train_acer_ale.py n_processes rom [options]
python train_nsq_ale.py n_processes rom [options]
python train_dqn_ale.py rom [options]
```

Specify `--help` or read code for options.
