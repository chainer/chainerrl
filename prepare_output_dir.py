from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import os
import tempfile
import json
import subprocess
import argparse
import datetime


def prepare_output_dir(args, user_specified_dir=None):
    """Prepare output directory.

    An output directory is created if it does not exist. Then the following
    infomation is saved into the directory:
      args.txt: command-line arguments
      git-status.txt: result of `git status`
      git-log.txt: result of `git log`
      git-diff.txt: result of `git diff`

    Args:
      args: dict that describes command-line arguments
      user_specified_dir: directory path
    """
    time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        outdir = os.path.join(user_specified_dir, time_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=time_str)

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ)))

    # Save `git status`
    with open(os.path.join(outdir, 'git-status.txt'), 'w') as f:
        f.write(subprocess.getoutput('git status'))

    # Save `git log`
    with open(os.path.join(outdir, 'git-log.txt'), 'w') as f:
        f.write(subprocess.getoutput('git log'))

    # Save `git diff`
    with open(os.path.join(outdir, 'git-diff.txt'), 'w') as f:
        f.write(subprocess.getoutput('git diff'))

    return outdir
