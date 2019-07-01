#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/drqn
python examples/atari/train_drqn_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/ale/dqn --gpu $gpu --recurrent --flicker
model=$(find $outdir/ale/dqn -name "*_finish")
python examples/atari/train_drqn_ale.py --env PongNoFrameskip-v4 --demo --load $model --demo-n-episodes 1 --max-frames 50 --outdir $outdir/temp --gpu $gpu --recurrent --flicker