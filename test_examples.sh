#!/bin/bash

set -Ceu

outdir=$(mktemp -d)
echo "outdir: $outdir"

gpu="$1"

# ale/dqn
python examples/ale/train_dqn_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/ale/dqn --gpu $gpu
model=$(find $outdir/ale/dqn -name "*_finish")
python examples/ale/train_dqn_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# atari/dqn
python examples/atari/dqn/train_dqn.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/dqn --eval-n-steps 200 --eval-interval 50 --n-best-episodes 1 --gpu $gpu
model=$(find $outdir/atari/dqn -name "*_finish")
python examples/atari/dqn/train_dqn.py --env PongNoFrameskip-v4 --demo --load $model --outdir $outdir/temp --eval-n-steps 200 --gpu $gpu

# ale/dqn batch
python examples/ale/train_dqn_batch_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/ale/dqn_batch --gpu $gpu
model=$(find $outdir/ale/dqn_batch -name "*_finish")
python examples/ale/train_dqn_batch_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# ale/a2c
python examples/ale/train_a2c_ale.py --env PongNoFrameskip-v4 --steps 100 --update-steps 50 --outdir $outdir/ale/a2c
model=$(find $outdir/ale/a2c -name "*_finish")
python examples/ale/train_a2c_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1

# ale/a3c
if [[ $gpu -lt 0 ]]; then
  python examples/ale/train_a3c_ale.py 4 --env PongNoFrameskip-v4 --steps 100 --outdir $outdir/ale/a3c
  model=$(find $outdir/ale/a3c -name "*_finish")
  python examples/ale/train_a3c_ale.py 4 --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp
fi

# ale/acer
if [[ $gpu -lt 0 ]]; then
  python examples/ale/train_acer_ale.py 4 --env PongNoFrameskip-v4 --steps 100 --outdir $outdir/ale/acer
  model=$(find $outdir/ale/acer -name "*_finish")
  python examples/ale/train_acer_ale.py 4 --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp
fi

# ale/nsq
if [[ $gpu -lt 0 ]]; then
  python examples/ale/train_nsq_ale.py 4 --env PongNoFrameskip-v4 --steps 100 --outdir $outdir/ale/nsq
  model=$(find $outdir/ale/nsq -name "*_finish")
  python examples/ale/train_nsq_ale.py 4 --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp
fi

# ale/ppo
python examples/ale/train_ppo_ale.py --env PongNoFrameskip-v4 --steps 100 --update-interval 50 --batchsize 16 --epochs 2 --outdir $outdir/ale/ppo --gpu $gpu
model=$(find $outdir/ale/ppo -name "*_finish")
python examples/ale/train_ppo_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# ale/categorical_dqn
python examples/ale/train_categorical_dqn_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/ale/categorical_dqn --gpu $gpu
model=$(find $outdir/ale/categorical_dqn -name "*_finish")
python examples/ale/train_categorical_dqn_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# ale/iqn
python examples/ale/train_iqn_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/ale/iqn --gpu $gpu
model=$(find $outdir/ale/iqn -name "*_finish")
python examples/ale/train_iqn_ale.py --env PongNoFrameskip-v4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# gym/dqn
python examples/gym/train_dqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/dqn --gpu $gpu
model=$(find $outdir/gym/dqn -name "*_finish")
python examples/gym/train_dqn_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# gym/a2c
python examples/gym/train_a2c_gym.py --steps 100 --update-steps 50 --outdir $outdir/gym/a2c --gpu $gpu
model=$(find $outdir/gym/a2c -name "*_finish")
python examples/gym/train_a2c_gym.py --demo --load $model --eval-n-runs 1 --gpu $gpu

# gym/a3c
python examples/gym/train_a3c_gym.py 4 --steps 100 --outdir $outdir/gym/a3c
model=$(find $outdir/gym/a3c -name "*_finish")
python examples/gym/train_a3c_gym.py 4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp

# gym/acer
python examples/gym/train_acer_gym.py 4 --steps 100 --outdir $outdir/gym/acer
model=$(find $outdir/gym/acer -name "*_finish")
python examples/gym/train_acer_gym.py 4 --demo --load $model --eval-n-runs 1 --outdir $outdir/temp

# gym/pcl
python examples/gym/train_pcl_gym.py --steps 100 --batchsize 2 --replay-start-size 2 --outdir $outdir/gym/pcl --gpu $gpu
model=$(find $outdir/gym/pcl -name "*_finish")
python examples/gym/train_pcl_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# gym/ddpg (specify non-mujoco env to test without mujoco)
python examples/gym/train_ddpg_gym.py --steps 100 --replay-start-size 50 --minibatch-size 32 --outdir $outdir/gym/ddpg --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/gym/ddpg -name "*_finish")
python examples/gym/train_ddpg_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu

# gym/reinforce
python examples/gym/train_reinforce_gym.py --steps 100 --batchsize 1 --outdir $outdir/gym/reinforce --gpu $gpu
model=$(find $outdir/gym/reinforce -name "*_finish")
python examples/gym/train_reinforce_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# gym/ppo (specify non-mujoco env to test without mujoco)
python examples/gym/train_ppo_gym.py --steps 100 --update-interval 50 --batchsize 16 --epochs 2 --outdir $outdir/gym/ppo --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/gym/ppo -name "*_finish")
python examples/gym/train_ppo_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu

# gym/ppo batch (specify non-mujoco env to test without mujoco)
python examples/gym/train_ppo_batch_gym.py --steps 100 --update-interval 50 --batchsize 16 --epochs 2 --outdir $outdir/gym/ppo_batch --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/gym/ppo_batch -name "*_finish")
python examples/gym/train_ppo_batch_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu

# gym/trpo (specify non-mujoco env to test without mujoco)
python examples/gym/train_trpo_gym.py --steps 100 --trpo-update-interval 50 --outdir $outdir/gym/trpo --env Pendulum-v0 --gpu $gpu
model=$(find $outdir/gym/trpo -name "*_finish")
python examples/gym/train_trpo_gym.py --demo --load $model --eval-n-runs 1 --env Pendulum-v0 --outdir $outdir/temp --gpu $gpu

# gym/categorical_dqn
python examples/gym/train_categorical_dqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/categorical_dqn --gpu $gpu
model=$(find $outdir/gym/categorical_dqn -name "*_finish")
python examples/gym/train_categorical_dqn_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# gym/iqn
python examples/gym/train_iqn_gym.py --steps 100 --replay-start-size 50 --outdir $outdir/gym/iqn --gpu $gpu
model=$(find $outdir/gym/iqn -name "*_finish")
python examples/gym/train_iqn_gym.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu

# grasping/dqn
python examples/grasping/train_dqn_batch_grasping.py --gpu $gpu --steps 100 --outdir $outdir/grasping/dqn
model=$(find $outdir/grasping/dqn -name "*_finish")
python examples/grasping/train_dqn_batch_grasping.py --demo --load $model --eval-n-runs 1 --outdir $outdir/temp --gpu $gpu
