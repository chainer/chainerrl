#!/bin/bash

run()
{
  dmux run -- python3 examples/ale/train_dqn_ale.py --env AsterixNoFrameskip-v4 --steps 10000000 --outdir results --arch nature $1 --seed $2 --final-exploration-frames 500000 --target-update-interval 1000 --replay-start-size 10000 --adam --eval-interval 10000 --noise-coef 1 --init-method /out
}

for num in {1..3}
do
#run "--agent SARSA" $num
run "--noisy-net-sigma 0.5 --agent SARSA --entropy-coef 1e-10 --buffer-size 10000" $num
run "--noisy-net-sigma 0.5 --agent SARSA --entropy-coef 1e-10 --buffer-size 100000" $num
run "--noisy-net-sigma 0.5 --agent SARSA --entropy-coef 1e-10 --buffer-size 1000000" $num
run "--noisy-net-sigma 0.5 --agent SARSA --entropy-coef 1e-13 --buffer-size 10000" $num
run "--noisy-net-sigma 0.5 --agent SARSA --entropy-coef 1e-13 --buffer-size 100000" $num
run "--noisy-net-sigma 0.5 --agent SARSA --entropy-coef 1e-13 --buffer-size 1000000" $num

done
