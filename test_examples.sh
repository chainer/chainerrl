#!/bin/bash

set -Ceu

python examples/ale/train_dqn_ale.py pong --steps 100 --replay-start-size 50
python examples/ale/train_a3c_ale.py 4 pong --steps 100
python examples/ale/train_nsq_ale.py 4 pong --steps 100

python examples/gym/train_dqn_gym.py --steps 100 --replay-start-size 50
python examples/gym/train_a3c_gym.py 4 --steps 100
python examples/gym/train_ddpg_gym.py --steps 100 --replay-start-size 50 --minibatch-size 32
