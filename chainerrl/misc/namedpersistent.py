from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer


def _namedchildren(link):
    if isinstance(link, chainer.Chain):
        for name in sorted(link._children):
            yield name, link.__dict__[name]
    elif isinstance(link, chainer.ChainList):
        for idx, child in enumerate(link._children):
            yield str(idx), child


def namedpersistent(link):
    """Return a generator of all (path, persistent) pairs for a given link.

    This function is adopted from https://github.com/chainer/chainer/pull/6788.
    Once it is merged into Chainer, we should use the property instead.

    Args:
        link (chainer.Link): Link.

    Returns:
        A generator object that generates all (path, persistent) pairs.
        The paths are relative from this link.
    """
    d = link.__dict__
    for name in sorted(link._persistent):
        yield '/' + name, d[name]
    for name, child in _namedchildren(link):
        prefix = '/' + name
        for path, persistent in namedpersistent(child):
            yield prefix + path, persistent
