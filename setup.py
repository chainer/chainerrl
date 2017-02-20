from setuptools import find_packages
from setuptools import setup
import sys

install_requires = [
    'atari_py',
    'cached-property',
    'chainer>=1.20.0.1',
    'future',
    'gym>=0.7.3',
    'numpy>=1.10.4',
    'pillow',
    'scipy',
]

test_requires = [
    'nose',
]

if sys.version_info < (3, 2):
    install_requires.append('fastcache')

if sys.version_info < (3, 4):
    install_requires.append('statistics')

if sys.version_info < (3, 5):
    install_requires.append('funcsigs')

setup(name='chainerrl',
      version='0.0.1',
      description='ChainerRL, a deep reinforcement learning library',
      author='Yasuhiro Fujita',
      author_email='fujita@preferred.jp',
      license='MIT License',
      packages=find_packages(),
      install_requires=install_requires,
      test_requires=test_requires)
