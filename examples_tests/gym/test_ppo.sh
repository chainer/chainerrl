#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/ppo (specify non-mujoco env to test without mujoco)
python examples/gym/train_ppo_gym.py --steps 100 --update-interval 50 --batchsize 16 --epochs 2 --outdir $outdir/gym/ppo --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/gym/ppo -name "*_finish")
python examples/gym/train_ppo_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu
