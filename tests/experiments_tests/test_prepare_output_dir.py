from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

from chainer import testing

import chainerrl


@contextlib.contextmanager
def work_dir(dirname):
    orig_dir = os.getcwd()
    os.chdir(dirname)
    yield
    os.chdir(orig_dir)


class TestIsUnderGitControl(unittest.TestCase):

    def test(self):

        tmp = tempfile.mkdtemp()

        # Not under git control
        with work_dir(tmp):
            self.assertFalse(chainerrl.experiments.is_under_git_control())

        # Run: git init
        with work_dir(tmp):
            subprocess.call(['git', 'init'])

        # Under git control
        with work_dir(tmp):
            self.assertTrue(chainerrl.experiments.is_under_git_control())


@testing.parameterize(*testing.product({
    'git': [True, False],
    'user_specified_dir': [tempfile.mkdtemp(), None],
    'argv': [['command', '--option'], None],
    'time_format': ['%Y%m%dT%H%M%S.%f', '%Y%m%d'],
}))
class TestPrepareOutputDir(unittest.TestCase):

    def test_prepare_output_dir(self):

        tmp = tempfile.mkdtemp()
        args = dict(a=1, b='2')
        os.environ['CHAINERRL_TEST_PREPARE_OUTPUT_DIR'] = 'T'

        with work_dir(tmp):

            if self.git:
                subprocess.call(['git', 'init'])
                with open('not_utf-8.txt', 'wb') as f:
                    f.write(b'\x80')
                subprocess.call(['git', 'add', 'not_utf-8.txt'])
                subprocess.call(['git', 'commit', '-m' 'commit1'])
                with open('not_utf-8.txt', 'wb') as f:
                    f.write(b'\x81')

            dirname = chainerrl.experiments.prepare_output_dir(
                args,
                user_specified_dir=self.user_specified_dir,
                argv=self.argv)

            self.assertTrue(os.path.isdir(dirname))

            if self.user_specified_dir:
                dirname.startswith(self.user_specified_dir)

            # args.txt
            args_path = os.path.join(dirname, 'args.txt')
            with open(args_path, 'r') as f:
                obj = json.load(f)
                self.assertEqual(obj, args)

            # environ.txt
            environ_path = os.path.join(dirname, 'environ.txt')
            with open(environ_path, 'r') as f:
                obj = json.load(f)
                self.assertEqual(obj['CHAINERRL_TEST_PREPARE_OUTPUT_DIR'], 'T')

            # command.txt
            command_path = os.path.join(dirname, 'command.txt')
            with open(command_path, 'r') as f:
                if self.argv:
                    self.assertTrue(' '.join(self.argv), f.read())
                else:
                    self.assertTrue(' '.join(sys.argv), f.read())

            for gitfile in ['git-head.txt',
                            'git-status.txt',
                            'git-log.txt',
                            'git-diff.txt']:
                if self.git:
                    self.assertTrue(os.path.exists(
                        os.path.join(dirname, gitfile)))
                else:
                    self.assertFalse(os.path.exists(
                        os.path.join(dirname, gitfile)))
