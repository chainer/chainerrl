#!/bin/bash
common="--env MountainCar-v0 --reward-scale-factor 1 --steps 400000 --action-repeat 4"

for num in {1..10}
do
dmux run -- python3 examples/gym/train_dqn_gym.py --final-exploration-steps 1000 $common --seed $num
dmux run -- python3 examples/gym/train_dqn_gym.py --final-exploration-steps 10000 $common --seed $num
dmux run -- python3 examples/gym/train_dqn_gym.py --final-exploration-steps 100000 $common --seed $num
dmux run -- python3 examples/gym/train_dqn_gym.py $common --noisy-net-sigma 0.5 --seed $num
dmux run -- python3 examples/gym/train_dqn_gym.py $common --noisy-net-sigma 0.5 --noise-constant 0.1 --seed $num
#dmux run -- python3 examples/gym/train_categorical_dqn_gym.py --env MountainCar-v0 --final-exploration-steps 100000 $common --seed $num
#dmux run -- python3 examples/gym/train_categorical_dqn_gym.py --env MountainCar-v0 $common --noisy-net-sigma 0.5 --seed $num
#dmux run -- python3 examples/gym/train_categorical_dqn_gym.py --env MountainCar-v0 $common --noisy-net-sigma 0.5 --noise-constant 0.1 --seed $num
done
