from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import chainer


class Branched(chainer.ChainList):
    """Link that calls forward functions of child links in parallel.

    When either the `__call__` method of this link are called, all the
    argeuments are forwarded to each child link's `__call__` method.

    The returned values from the child links are returned as a tuple.

    Args:
        *links: Child links. Each link should be callable.
    """

    def __call__(self, *args, **kwargs):
        """Forward the arguments to the child links.

        Args:
            *args, **kwargs: Any arguments forwarded to child links. Each child
                link should be able to accept the arguments.

        Returns:
            tuple: Tuple of the returned values from the child links.
        """
        return tuple(link(*args, **kwargs) for link in self)
