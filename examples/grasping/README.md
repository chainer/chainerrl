# Bullet-based robotic grasping

This directory contains example scripts that learn to grasp objects in an environment simulated by Bullet, a physics simulator.

![Grasping](../../assets/grasping.gif)

## Files

- `train_dqn_batch_grasping.py`: DoubleDQN + prioritized experience replay

## Requirements

- pybullet>=2.1.2

## How to run

```
# Train with one simulator, which is slow

python examples/pybullet/train_dqn_batch_grasping.py

# Train with 96 simulators run in parallel, which is faster

python examples/pybullet/train_dqn_batch_grasping.py --num-envs 96

# Watch how the learned agent performs

python examples/pybullet/train_dqn_batch_grasping.py --demo --render --load <path to agent>
```
