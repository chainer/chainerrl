#!/bin/bash

run()
{
  common="examples/gym/train_dqn_gym.py --env MountainCar-v0 --steps 400000 --action-repeat 1 --gamma 0.99 --n-hidden-layers 1 --n-hidden-channels 16"

  for num in {1..5}
  do
  #dmux run -- python3 examples/gym/train_dqn_gym.py --final-exploration-steps 1000 $common --seed $num
  #dmux run -- python3 examples/gym/train_dqn_gym.py --final-exploration-steps 10000 $common --seed $num
  dmux run -- python3 $common --final-exploration-steps 100000 --seed $num --reward-scale-factor 1

  #dmux run -- python3 $common --noisy-net-sigma 0.3 --seed $num
  #dmux run -- python3 $common --noisy-net-sigma 0.5 --seed $num
  #dmux run -- python3 $common --noisy-net-sigma 0.7 --seed $num

  #dmux run -- python3 $common --noisy-net-sigma 100 --noise-constant 0.3 --seed $num
  #dmux run -- python3 $common --noisy-net-sigma 100 --noise-constant 0.5 --seed $num
  #dmux run -- python3 $common --noisy-net-sigma 100 --noise-constant 0.7 --seed $num

  #dmux run -- python3 examples/gym/train_categorical_dqn_gym.py --env MountainCar-v0 --final-exploration-steps 100000 $common --seed $num
  #dmux run -- python3 examples/gym/train_categorical_dqn_gym.py --env MountainCar-v0 $common --noisy-net-sigma 0.5 --seed $num
  #dmux run -- python3 examples/gym/train_categorical_dqn_gym.py --env MountainCar-v0 $common --noisy-net-sigma 0.5 --noise-constant 0.1 --seed $num
  done
}

run 1 0.001
#run 0.1 0.001
#run 1 0.003
#run 1 0.0003
#run 1 0.0001
#run 1 0.00003
#run 0.1 0.003
#run "--target-update-method soft --soft-update-tau 0.125"
#run "--rbuf 1000"
#run "--rbuf 2000"
#run 0.1 0.00003

#run 1 0.001 10
#run 1 0.001 1
