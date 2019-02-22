FROM ubuntu:18.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python2.7-dev python-pip zlib1g-dev make cmake g++ git \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN python2 -m pip install --no-cache-dir setuptools
RUN python2 -m pip install --no-cache-dir \
    chainer pytest-cov numpy scipy cached-property future gym==0.10.11 pillow \
    jupyter atari_py==0.1.1 autopep8 flake8 coveralls \
    opencv-python pybullet mock fastcache funcsigs statistics

RUN git config --global user.email "you@example.com"
RUN git config --global user.name "Your Name"
