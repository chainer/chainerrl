#!/bin/bash

run()
{
  for num in {1..5}
  do
  dmux run -- python3 examples/ale/train_dqn_ale.py --env AsterixNoFrameskip-v4 --steps 50000000 --outdir results --seed $num --arch dueling $1
  done
}

#run "--noisy-net-sigma -1"
#run "--noisy-net-sigma 0.5"
run "--noisy-net-sigma 0.5 --prop --agent DQN"
run "--noisy-net-sigma 0.5 --prop --agent DDQN"
run "--noisy-net-sigma 0.5 --prop --agent SARSA"
run "--noisy-net-sigma 0.5 --agent DQN"
run "--noisy-net-sigma 0.5 --agent DDQN"
run "--noisy-net-sigma 0.5 --agent SARSA"
#run "--noisy-net-sigma 100 --noise-constant 0.5"
