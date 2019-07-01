#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# mujoco/ddpg_batch (specify non-mujoco env to test without mujoco)
python examples/mujoco/train_ddpg_batch_gym.py --steps 100 --replay-start-size 50 --minibatch-size 32 --outdir $outdir/mujoco/ddpg_batch --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/mujoco/ddpg_batch -name "*_finish")
python examples/mujoco/train_ddpg_batch_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu
