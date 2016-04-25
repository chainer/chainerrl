#!/bin/sh

set -e

# Install ALE
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
make -j 4
pip install --user .

# Get ROMs
cd ..
wget http://www.atarimania.com/roms/Roms.zip
unzip Roms.zip
unzip ROMS.zip
ln -s ROMS/Video\ Olympics\ -\ Pong\ Sports\ \(Paddle\)\ \(1977\)\ \(Atari\,\ Joe\ Decuir\ -\ Sears\)\ \(CX2621\ -\ 99806\,\ 6-99806\,\ 49-75104\)\ ~.bin pong.bin
ln -s ROMS/Breakout\ -\ Breakaway\ IV\ \(Paddle\)\ \(1978\)\ \(Atari\,\ Brad\ Stewart\ -\ Sears\)\ \(CX2622\ -\ 6-99813\,\ 49-75107\)\ ~.bin breakout.bin
