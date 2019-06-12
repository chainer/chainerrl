#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/pcl
python examples/gym/train_pcl_gym.py --steps 100 --batchsize 2 --replay-start-size 2 --outdir $outdir/gym/pcl --gpu $gpu
model=$(find $outdir/gym/pcl -name "*_finish")
python examples/gym/train_pcl_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu
