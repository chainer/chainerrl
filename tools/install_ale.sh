#!/bin/bash

set -eu

if ! hash git 2>/dev/null; then
  echo "You need git"
  exit 1
fi

if ! hash cmake 2>/dev/null; then
  echo "You need cmake"
  exit 1
fi

if ! hash unzip 2>/dev/null; then
  echo "You need unzip"
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

# Get ROMs
cd ..
if [ ! -e Roms.zip ]; then
  wget http://www.atarimania.com/roms/Roms.zip
fi
if [ ! -e ROMS.zip ]; then
  unzip Roms.zip
fi
if [ ! -e ROMS ]; then
  unzip ROMS.zip
fi
if [ ! -e pong.bin ]; then
  ln -s ROMS/Video\ Olympics\ -\ Pong\ Sports\ \(Paddle\)\ \(1977\)\ \(Atari\,\ Joe\ Decuir\ -\ Sears\)\ \(CX2621\ -\ 99806\,\ 6-99806\,\ 49-75104\)\ ~.bin pong.bin
fi
if [ ! -e breakout.bin ]; then
  ln -s ROMS/Breakout\ -\ Breakaway\ IV\ \(Paddle\)\ \(1978\)\ \(Atari\,\ Brad\ Stewart\ -\ Sears\)\ \(CX2622\ -\ 6-99813\,\ 49-75107\)\ ~.bin breakout.bin
fi
if [ ! -e seaquest.bin ]; then
  ln -s ROMS/Seaquest\ \(1983\)\ \(Activision\,\ Steve\ Cartwright\)\ \(AX-022\)\ ~.bin seaquest.bin
fi
