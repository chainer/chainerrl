#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/a2c
python examples/gym/train_a2c_gym.py --steps 100 --update-steps 50 --outdir $outdir/gym/a2c --gpu $gpu
model=$(find $outdir/gym/a2c -name "*_finish")
python examples/gym/train_a2c_gym.py --demo --load $model --eval-n-runs 1 --gpu $gpu
