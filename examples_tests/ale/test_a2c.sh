#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# ale/a2c
python examples/ale/train_a2c_ale.py --env PongNoFrameskip-v4 --steps 100 --update-steps 50 --outdir $outdir/ale/a2c
model=$(find $outdir/ale/a2c -name "*_finish")
python examples/ale/train_a2c_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1
