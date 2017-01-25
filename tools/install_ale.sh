#!/bin/bash

# This script will clone Arcade-Learning-Environment to the current directory,
# build it and install ale_python_interface.


set -Ceu

if ! hash git 2>/dev/null; then
  echo "You need git"
  exit 1
fi

if ! hash cmake 2>/dev/null; then
  echo "You need cmake"
  exit 1
fi

# Install ALE
if [ ! -e Arcade-Learning-Environment ]; then
  git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
fi
cd Arcade-Learning-Environment
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
make -j 4
pip install --user .
