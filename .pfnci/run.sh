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
: "${SLOW:=0}"

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
  if (( !SLOW )); then
    marker='not slow'
  else
    marker='slow'
  fi
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

  UBUNTU_VERSION_ID=$(grep DISTRIB_RELEASE /etc/lsb-release | cut -d "=" -f2)
  if [ "$UBUNTU_VERSION_ID" = "16.04" ]; then
    # Because ffmpeg of ubuntu 16.04 causes segmentation fault,
    # we use jonathonf/ffmpeg-3
    apt-get update -q
    apt-get install -qy --no-install-recommends software-properties-common
    add-apt-repository ppa:cran/ffmpeg-3
  fi

  apt-get update -q
  apt-get install -qy --no-install-recommends \
      "${PYTHON}-dev" "${PYTHON}-pip" "${PYTHON}-setuptools" \
      zlib1g-dev make cmake g++ git ffmpeg freeglut3-dev xvfb

  if [ "${CHAINER}" != '' ]; then
    "${PYTHON}" -m pip install "chainer==${CHAINER}"
  fi

  if [ "${PYTHON}" == 'python' ]; then
    "${PYTHON}" -m pip install \
        'cython==0.28.0' 'numpy<1.10' 'scipy<0.19' 'more-itertools<=5.0.0'
  fi

  "${PYTHON}" -m pip install /chainerrl
  # TODO(chainerrl): Prepare test target instead.
  # pytest does not run with attrs==19.2.0 (https://github.com/pytest-dev/pytest/issues/3280)  # NOQA
  "${PYTHON}" -m pip install \
      'pytest==4.1.1' 'attrs==19.1.0' 'pytest-xdist==1.26.1' \
      'atari_py==0.1.1' 'opencv-python' 'zipp==1.0.0'

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

  # Xvfb's default screen is 1280x1024x8, which seems to cause a problem.
  # https://bugzilla.redhat.com/show_bug.cgi?id=904851
  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 \
      xvfb-run --server-args="-screen 0 1280x800x24" \
      xpytest "${xpytest_args[@]}" '/chainerrl/tests/**/test_*.py'
}

main
