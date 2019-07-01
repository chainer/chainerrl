#!/bin/bash

set -Ceu

outdir=$(mktemp -d)

gpu="$1"

# Chainer 4 does not support open_pickle_dataset_writer, which is used by the demonstration collection example.
pickle_writer_support=$(python -c "import chainer; from distutils.version import StrictVersion; print(1 if StrictVersion(chainer.__version__) >= StrictVersion('5.0.0') else 0)")

# atari/collect_demos
if [[ $pickle_writer_support = 1 ]]; then
	python examples/atari/train_dqn_ale.py --env PongNoFrameskip-v4 --steps 100 --replay-start-size 50 --outdir $outdir/atari/dqn --gpu $gpu
	model=$(find $outdir/atari/dqn -name "*_finish")
	python examples/atari/collect_demos_ale.py --env PongNoFrameskip-v4 --load $model --steps 100 --outdir $outdir/temp --gpu $gpu
fi
