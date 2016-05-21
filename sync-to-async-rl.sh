#!/bin/bash

set -u

sync() {
  cp $1 ../async-rl/$1
}

sync a3c_ale.py
sync a3c.py
sync ale.py
sync async.py
sync copy_param.py
sync demo_a3c_ale.py
sync dqn_head.py
sync dqn_phi.py
sync environment.py
sync nonbias_weight_decay.py
sync plot_scores.py
sync policy_output.py
sync policy.py
sync prepare_output_dir.py
sync random_seed.py
sync rmsprop_async.py
sync v_function.py
sync init_like_torch.py
sync run_a3c.py
sync train_a3c_doom.py
sync doom_env.py
sync demo_a3c_doom.py
