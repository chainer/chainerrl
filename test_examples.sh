#!/bin/bash

set -Ceu

outdir=$(mktemp -d)
echo "outdir: $outdir"

# ale/dqn
python examples/ale/train_dqn_ale.py pong --steps 100 --replay-start-size 50 --outdir $outdir/ale/dqn
model=$(find $outdir/ale/dqn -name "*_finish")
python examples/ale/train_dqn_ale.py pong --demo --load $model --eval-n-runs 1

# ale/a3c
python examples/ale/train_a3c_ale.py 4 pong --steps 100 --outdir $outdir/ale/a3c
model=$(find $outdir/ale/a3c -name "*_finish")
python examples/ale/train_a3c_ale.py 4 pong --demo --load $model --eval-n-runs 1

# ale/nsq
python examples/ale/train_nsq_ale.py 4 pong --steps 100 --outdir $outdir/ale/nsq
model=$(find $outdir/ale/nsq -name "*_finish")
python examples/ale/train_nsq_ale.py 4 pong --demo --load $model --eval-n-runs 1

# gym/dqn
python examples/gym/train_dqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/dqn
model=$(find $outdir/gym/dqn -name "*_finish")
python examples/gym/train_dqn_gym.py --demo --load $model --eval-n-runs 1

# gym/a3c
python examples/gym/train_a3c_gym.py 4 --steps 100 --outdir $outdir/gym/a3c
model=$(find $outdir/gym/a3c -name "*_finish")
python examples/gym/train_a3c_gym.py 4 --demo --load $model --eval-n-runs 1

# gym/ddpg
python examples/gym/train_ddpg_gym.py --steps 100 --replay-start-size 50 --minibatch-size 32 --outdir $outdir/gym/ddpg
model=$(find $outdir/gym/ddpg -name "*_finish")
python examples/gym/train_ddpg_gym.py --demo --load $model --eval-n-runs 1
