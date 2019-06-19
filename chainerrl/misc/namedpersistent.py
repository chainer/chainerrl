from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer


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
    if isinstance(link, chainer.Chain):
        for name in sorted(link._children):
            prefix = '/' + name
            for path, persistent in namedpersistent(d[name]):
                yield prefix + path, persistent
    elif isinstance(link, chainer.ChainList):
        for idx, link in enumerate(link._children):
            prefix = '/{}'.format(idx)
            for path, persistent in namedpersistent(link):
                yield prefix + path, persistent
