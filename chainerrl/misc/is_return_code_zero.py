from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import os
import subprocess


def is_return_code_zero(args):
    """Return true iff the given command's return code is zero.

    All the messages to stdout or stderr are suppressed.
    """
    with open(os.devnull, 'wb') as FNULL:
        try:
            subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        except subprocess.CalledProcessError:
            # The given command returned an error
            return False
        except OSError:
            # The given command was not found
            return False
        return True
