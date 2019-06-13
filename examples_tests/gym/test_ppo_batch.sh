#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/ppo batch (specify non-mujoco env to test without mujoco)
python examples/gym/train_ppo_batch_gym.py --steps 100 --update-interval 50 --batchsize 16 --epochs 2 --outdir $outdir/gym/ppo_batch --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/gym/ppo_batch -name "*_finish")
python examples/gym/train_ppo_batch_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu
