============
Installation
============

How to install ChainerRL
========================

ChainerRL is tested with Python 2.7+ and 3.5.1+. For other requirements, see ``requirements.txt``.

.. literalinclude:: ../requirements.txt
  :caption: requirements.txt

ChainerRL can be installed via PyPI:

::

 pip install chainerrl

It can also be installed from the source code:

::

 python setup.py install

For Windows users
=================

ChainerRL contains ``atari_py`` as dependencies, and windows users may face errors while installing it.
This problem is discussed in `OpenAI gym issues <https://github.com/openai/gym/issues/11>`_,
and one possible counter measure is to enable "Bash on Ubuntu on Windows" for Windows 10 users.

Refer `Official install guilde <https://msdn.microsoft.com/en-us/commandline/wsl/install_guide>`_ to install "Bash on Ubuntu on Windows".
