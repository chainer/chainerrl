#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# ale/dqn batch
python examples/ale/train_dqn_batch_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/ale/dqn_batch --gpu $gpu
model=$(find $outdir/ale/dqn_batch -name "*_finish")
python examples/ale/train_dqn_batch_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu
