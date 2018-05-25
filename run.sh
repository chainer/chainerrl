#!/bin/bash
for num in {1..10}
do
#dmux run -- python3 examples/gym/train_dqn_gym.py --env MountainCar-v0 --final-exploration-steps 100000 --steps 400000 --reward-scale-factor 1 --seed $num
#dmux run -- python3 examples/gym/train_dqn_gym.py --env MountainCar-v0 --final-exploration-steps 100000 --steps 400000 --reward-scale-factor 1 --noisy-net-sigma 0.5 --seed $num
#dmux run -- python3 examples/gym/train_dqn_gym.py --env MountainCar-v0 --final-exploration-steps 100000 --steps 400000 --reward-scale-factor 1 --noisy-net-sigma 0.5 --noise-constant 0.1 --seed $num
dmux run -- python3 examples/gym/train_categorical_dqn_gym.py --env MountainCar-v0 --final-exploration-steps 100000 --steps 400000 --reward-scale-factor 1 --seed $num
dmux run -- python3 examples/gym/train_categorical_dqn_gym.py --env MountainCar-v0 --final-exploration-steps 100000 --steps 400000 --reward-scale-factor 1 --noisy-net-sigma 0.5 --seed $num
dmux run -- python3 examples/gym/train_categorical_dqn_gym.py --env MountainCar-v0 --final-exploration-steps 100000 --steps 400000 --reward-scale-factor 1 --noisy-net-sigma 0.5 --noise-constant 0.1 --seed $num
done
