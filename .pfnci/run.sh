#!/bin/bash
# run.sh is a script to run unit tests inside Docker.  This should be injected
# by and called from script.sh.
#
# Usage: run.sh [target]
# - target is a test target (e.g., "py37").
#
# Environment variables:
# - GPU (default: 0) ... Set a number of GPUs to GPU.
#       CAVEAT: Setting GPU>0 disables non-GPU tests, and setting GPU=0 disables
#               GPU tests.

set -eux

cp -a /src /chainerrl
mkdir -p /chainerrl/.git
cd /

# Remove pyc files.  When the CI is triggered with a user's local files, pyc
# files generated on the user's local machine and they often cause failures.
find /chainerrl -name "*.pyc" -exec rm -f {} \;

TARGET="$1"
: "${GPU:=0}"
: "${XPYTEST_NUM_THREADS:=$(nproc)}"
: "${PYTHON=python3}"
: "${CHAINER=}"

# Use multi-process service to prevent GPU flakiness caused by running many
# processes on a GPU.  Specifically, it seems running more than 16 processes
# sometimes causes "cudaErrorLaunchFailure: unspecified launch failure".
if (( GPU > 0 )); then
    nvidia-smi -c EXCLUSIVE_PROCESS
    nvidia-cuda-mps-control -d
fi

################################################################################
# Main function
################################################################################

main() {
  marker='not slow'
  if (( !GPU )); then
    marker+=' and not gpu'
    bucket=1
  else
    marker+=' and gpu'
    bucket="${GPU}"
  fi

  xpytest_args=(
      --python="${PYTHON}" -m "${marker}"
      --bucket="${bucket}" --thread="$(( XPYTEST_NUM_THREADS / bucket ))"
      --hint="/chainerrl/.pfnci/hint.pbtxt"
  )

  apt-get update -q
  apt-get install -qy --no-install-recommends \
      "${PYTHON}-dev" "${PYTHON}-pip" "${PYTHON}-setuptools" \
      zlib1g-dev make cmake g++ git

  if [ "${CHAINER}" != '' ]; then
    "${PYTHON}" -m pip install "chainer==${CHAINER}"
  fi

  if [ "${PYTHON}" == 'python' ]; then
    "${PYTHON}" -m pip install \
        'cython==0.28.0' 'numpy<1.10' 'scipy<0.19' 'more-itertools<=5.0.0'
  fi

  "${PYTHON}" -m pip install /chainerrl
  # TODO(chainerrl): Prepare test target instead.
  "${PYTHON}" -m pip install \
      'pytest==4.1.1' 'pytest-xdist==1.26.1' 'mock' \
      'atari_py==0.1.1' 'opencv-python'

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 \
      xpytest "${xpytest_args[@]}" '/chainerrl/tests/**/test_*.py'
}

main
