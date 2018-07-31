from setuptools import find_packages
from setuptools import setup
import sys

gym_require = 'gym>=0.7.3'
if sys.version_info < (3, 0):
    gym_require += ',!=0.9.6'

install_requires = [
    'cached-property',
    'chainer>=2.0.0',
    'future',
    gym_require,
    'numpy>=1.10.4',
    'pillow',
    'scipy',
]

test_requires = [
    'pytest',
]

if sys.version_info < (3, 2):
    install_requires.append('fastcache')

if sys.version_info < (3, 4):
    install_requires.append('statistics')

if sys.version_info < (3, 5):
    install_requires.append('funcsigs')

setup(name='chainerrl',
      version='0.3.0',
      description='ChainerRL, a deep reinforcement learning library',
      author='Yasuhiro Fujita',
      author_email='fujita@preferred.jp',
      license='MIT License',
      packages=find_packages(),
      install_requires=install_requires,
      test_requires=test_requires)
