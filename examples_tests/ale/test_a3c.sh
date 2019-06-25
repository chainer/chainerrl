#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# ale/a3c (only for cpu)
if [[ $gpu -lt 0 ]]; then
  python examples/ale/train_a3c_ale.py 4 --env PongNoFrameskip-v4 --steps 100 --outdir $outdir/ale/a3c
  model=$(find $outdir/ale/a3c -name "*_finish")
  python examples/ale/train_a3c_ale.py 4 --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp
fi
