import codecs
from setuptools import find_packages
from setuptools import setup
import sys

install_requires = [
    'cached-property',
    'chainer>=2.0.0',
    'future',
    'gym>=0.9.7',
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
      version='0.7.0',
      description='ChainerRL, a deep reinforcement learning library',
      long_description=codecs.open('README.md', 'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      author='Yasuhiro Fujita',
      author_email='fujita@preferred.jp',
      license='MIT License',
      packages=find_packages(),
      install_requires=install_requires,
      test_requires=test_requires)
