#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# atlas/soft_actor_critic
python examples/atlas/train_soft_actor_critic_atlas.py --gpu $gpu --steps 100 --outdir $outdir/atlas/soft_actor_critic
model=$(find $outdir/atlas/soft_actor_critic -name "*_finish")
python examples/atlas/train_soft_actor_critic_atlas.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu
