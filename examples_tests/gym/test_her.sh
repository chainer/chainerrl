#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/her (specify non-mujoco env to test without mujoco)
python examples/gym/train_her_gym.py --steps 100 --replay-start-size 50 --minibatch-size 32 --outdir $outdir/gym/her --env FetchPickAndPlace-v1 --gpu $gpu
model=$(find $outdir/gym/ddpg -name "*_finish")
python examples/gym/train_her_gym.py --demo --load $model --eval-n-runs 1 --env FetchPickAndPlace-v1 --outdir $outdir/temp --gpu $gpu