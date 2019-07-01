#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/iqn
python examples/gym/train_iqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/iqn --gpu $gpu
model=$(find $outdir/gym/iqn -name "*_finish")
python examples/gym/train_iqn_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu
