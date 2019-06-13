#!/bin/bash

set -Ceu

outdir=$(mktemp -d)
echo "outdir: $outdir"

gpu="$1"

# atari/rainbow
python examples/atari/rainbow/train_rainbow.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/rainbow --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu $gpu
model=$(find $outdir/atari/rainbow -name "*_finish")
python examples/atari/rainbow/train_rainbow.py --env PongNoFrameskip-v4 --demo --load $model --outdir $outdir/temp --eval-n-steps 200 --gpu $gpu
