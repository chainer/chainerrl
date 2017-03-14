#!/bin/bash

set -Ceu

outdir=$(mktemp -d)
echo "outdir: $outdir"

gpu="$1"

# ale/dqn
python examples/ale/train_dqn_ale.py pong --steps 100 --replay-start-size 50 --outdir $outdir/ale/dqn --gpu $gpu
model=$(find $outdir/ale/dqn -name "*_finish")
python examples/ale/train_dqn_ale.py pong --demo --load $model --eval-n-runs 1 --gpu $gpu

# ale/a3c
python examples/ale/train_a3c_ale.py 4 pong --steps 100 --outdir $outdir/ale/a3c
model=$(find $outdir/ale/a3c -name "*_finish")
python examples/ale/train_a3c_ale.py 4 pong --demo --load $model --eval-n-runs 1

# ale/acer
python examples/ale/train_acer_ale.py 4 pong --steps 100 --outdir $outdir/ale/acer
model=$(find $outdir/ale/acer -name "*_finish")
python examples/ale/train_acer_ale.py 4 pong --demo --load $model --eval-n-runs 1

# ale/nsq
python examples/ale/train_nsq_ale.py 4 pong --steps 100 --outdir $outdir/ale/nsq
model=$(find $outdir/ale/nsq -name "*_finish")
python examples/ale/train_nsq_ale.py 4 pong --demo --load $model --eval-n-runs 1

# gym/dqn
python examples/gym/train_dqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/dqn --gpu $gpu
model=$(find $outdir/gym/dqn -name "*_finish")
python examples/gym/train_dqn_gym.py --demo --load $model --eval-n-runs 1 --gpu $gpu

# gym/a3c
python examples/gym/train_a3c_gym.py 4 --steps 100 --outdir $outdir/gym/a3c
model=$(find $outdir/gym/a3c -name "*_finish")
python examples/gym/train_a3c_gym.py 4 --demo --load $model --eval-n-runs 1

# gym/acer
python examples/gym/train_acer_gym.py 4 --steps 100 --outdir $outdir/gym/acer
model=$(find $outdir/gym/acer -name "*_finish")
python examples/gym/train_acer_gym.py 4 --demo --load $model --eval-n-runs 1

# gym/pcl
python examples/gym/train_pcl_gym.py --steps 100 --outdir $outdir/gym/pcl --gpu $gpu
model=$(find $outdir/gym/pcl -name "*_finish")
python examples/gym/train_pcl_gym.py --demo --load $model --eval-n-runs 1 --gpu $gpu

# gym/ddpg (specify non-mujoco env to test without mujoco)
python examples/gym/train_ddpg_gym.py --steps 100 --replay-start-size 50 --minibatch-size 32 --outdir $outdir/gym/ddpg --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/gym/ddpg -name "*_finish")
python examples/gym/train_ddpg_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --gpu $gpu
