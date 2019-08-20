#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/nsq (only for cpu)
if [[ $gpu -lt 0 ]]; then
  python examples/atari/train_nsq_ale.py 4 --env PongNoFrameskip-v4 --steps 100 --outdir $outdir/atari/nsq
  model=$(find $outdir/atari/nsq -name "*_finish")
  python examples/atari/train_nsq_ale.py 4 --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp
fi
