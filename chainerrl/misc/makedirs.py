from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import os
import six


def makedirs(name, mode=0o777, exist_ok=False):
    """An wrapper of os.makedirs that accepts exist_ok."""
    if six.PY2:
        try:
            os.makedirs(name, mode)
        except OSError:
            if not os.path.isdir(name):
                raise
    else:
        os.makedirs(name, mode, exist_ok=exist_ok)
