# Examples for OpenAI Gym environments

- `train_a2c_gym.py`: A2C for both discrete action and continuous action spaces
- `train_a3c_gym.py`: A3C for both discrete action and continuous action spaces
- `train_acer_gym.py`: DiscreteACER for discrete action spaces
- `train_categorical_dqn_gym.py`: CategoricalDQN for discrete action action spaces
- `train_dqn_gym.py`: DQN for both discrete action and continuous action spaces
- `train_iqn_gym.py`: IQN for discrete action spaces
- `train_pcl_gym.py`: PCL for both discrete action and continuous action spaces
- `train_reinforce_gym.py`: REINFORCE for both discrete action and continuous action spaces (only for episodic envs)

## How to run

```
python train_a2c_gym.py [options]
python train_a3c_gym.py n_processes [options]
python train_acer_gym.py n_processes [options]
python train_categorical_dqn_gym.py [options]
python train_dqn_gym.py [options]
python train_iqn_gym.py [options]
python train_pcl_gym.py [options]
python train_reinforce_gym.py [options]
```

Specify `--help` or read code for options.
