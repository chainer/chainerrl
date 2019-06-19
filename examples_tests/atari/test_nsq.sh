#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# ale/nsq (only for cpu)
if [[ $gpu -lt 0 ]]; then
  python examples/ale/train_nsq_ale.py 4 --env PongNoFrameskip-v4 --steps 100 --outdir $outdir/ale/nsq
  model=$(find $outdir/ale/nsq -name "*_finish")
  python examples/ale/train_nsq_ale.py 4 --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp
fi
