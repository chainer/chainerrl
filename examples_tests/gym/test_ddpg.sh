#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/ddpg (specify non-mujoco env to test without mujoco)
python examples/gym/train_ddpg_gym.py --steps 100 --replay-start-size 50 --minibatch-size 32 --outdir $outdir/gym/ddpg --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/gym/ddpg -name "*_finish")
python examples/gym/train_ddpg_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu
