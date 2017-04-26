from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile


def is_return_code_zero(args):
    """Return true iff the given command's return code is zero.

    All the messages to stdout or stderr are suppressed.
    """
    FNULL = open(os.devnull, 'w')
    try:
        subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
    except subprocess.CalledProcessError:
        return False
    return True


def is_under_git_control():
    """Return true iff the current directory is under git control."""
    return is_return_code_zero(['git', 'rev-parse'])


def prepare_output_dir(args, user_specified_dir=None, argv=None,
                       time_format='%Y%m%dT%H%M%S.%f'):
    """Prepare a directory for outputting training results.

    An output directory, which ends with the current datetime string,
    is created. Then the following infomation is saved into the directory:

        args.txt: command line arguments
        command.txt: command itself
        environ.txt: environmental variables

    Additionally, if the current directory is under git control, the following
    information is saved:

        git-head.txt: result of `git rev-parse HEAD`
        git-status.txt: result of `git status`
        git-log.txt: result of `git log`
        git-diff.txt: result of `git diff`

    Args:
        args (dict or argparse.Namespace): Arguments to save
        user_specified_dir (str or None): If str is specified, the output
            directory is created under that path. If not specified, it is
            created as a new temporary directory instead.
        argv (list or None): The list of command line arguments passed to a
            script. If not specified, sys.argv is used instead.
        time_format (str): Format used to represent the current datetime. The
        default format is the basic format of ISO 8601.
    Returns:
        Path of the output directory created by this function (str).
    """
    time_str = datetime.datetime.now().strftime(time_format)
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

    # Save all the environment variables
    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ)))

    # Save the command
    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    if is_under_git_control():
        # Save `git rev-parse HEAD` (SHA of the current commit)
        with open(os.path.join(outdir, 'git-head.txt'), 'w') as f:
            f.write(subprocess.getoutput('git rev-parse HEAD'))

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
