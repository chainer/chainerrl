#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atari/drqn
python examples/atari/train_drqn_ale.py --env AsterixNoFrameskip-v4 --flicker --recurrent --steps 100 --replay-start-size 50 --outdir $outdir/atari/reproduction/dqn --eval-n-steps 200 --eval-interval 50 --gpu $gpu
model=$(find $outdir/atari/reproduction/dqn -name "*_finish")
python examples/atari/train_drqn_ale.py --env AsterixNoFrameskip-v4 --flicker --recurrent --demo --load $model --outdir $outdir/temp --eval-n-steps 200 --demo-n-episodes 1 --gpu $gpu
