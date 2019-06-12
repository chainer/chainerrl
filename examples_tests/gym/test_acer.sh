#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# gym/acer (only for cpu)
if [[ $gpu -lt 0 ]]; then
  python examples/gym/train_acer_gym.py 4 --steps 100 --outdir $outdir/gym/acer
  model=$(find $outdir/gym/acer -name "*_finish")
  python examples/gym/train_acer_gym.py 4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp
fi
