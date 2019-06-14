#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# mujoco/ddpg (specify non-mujoco env to test without mujoco)
python examples/mujoco/ddpg/train_ddpg.py --env Pendulum-v0 --gpu $gpu --steps 10 --replay-start-size 5 --batch-size 5 --outdir $outdir/mujoco/ddpg
model=$(find $outdir/mujoco/ddpg -name "*_finish")
python examples/mujoco/ddpg/train_ddpg.py --env Pendulum-v0 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu
