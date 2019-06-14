#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# ale/ppo
python examples/ale/train_ppo_ale.py --env PongNoFrameskip-v4 --steps 100 --update-interval 50 --batchsize 16 --epochs 2 --outdir $outdir/ale/ppo --gpu $gpu
model=$(find $outdir/ale/ppo -name "*_finish")
python examples/ale/train_ppo_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu
