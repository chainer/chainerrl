#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# ale/dqn
python examples/ale/train_dqn_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/ale/dqn --gpu $gpu
model=$(find $outdir/ale/dqn -name "*_finish")
python examples/ale/train_dqn_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu
