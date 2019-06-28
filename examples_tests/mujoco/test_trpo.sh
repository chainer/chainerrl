#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# mujoco/trpo (specify non-mujoco env to test without mujoco)
python examples/mujoco/train_trpo_gym.py --steps 100 --trpo-update-interval 50 --outdir $outdir/mujoco/trpo --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/mujoco/trpo -name "*_finish")
python examples/mujoco/train_trpo_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu
