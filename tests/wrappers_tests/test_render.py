from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import unittest

from chainer import testing
import mock

import chainerrl


@testing.parameterize(*testing.product({
    'render_kwargs': [
        {},
        {'mode': 'human'},
        {'mode': 'rgb_array'},
    ]
}))
class TestRender(unittest.TestCase):

    def test(self):
        orig_env = mock.Mock()
        # Reaches the terminal state after five actions
        orig_env.reset.side_effect = [
            ('state', 0),
            ('state', 3),
        ]
        orig_env.step.side_effect = [
            (('state', 1), 0, False, {}),
            (('state', 2), 1, True, {}),
        ]
        env = chainerrl.wrappers.Render(orig_env, **self.render_kwargs)

        # Not called env.render yet
        self.assertEqual(orig_env.render.call_count, 0)

        obs = env.reset()
        self.assertEqual(obs, ('state', 0))

        # Called once
        self.assertEqual(orig_env.render.call_count, 1)

        obs, reward, done, info = env.step(0)
        self.assertEqual(obs, ('state', 1))
        self.assertEqual(reward, 0)
        self.assertEqual(done, False)
        self.assertEqual(info, {})

        # Called twice
        self.assertEqual(orig_env.render.call_count, 2)

        obs, reward, done, info = env.step(0)
        self.assertEqual(obs, ('state', 2))
        self.assertEqual(reward, 1)
        self.assertEqual(done, True)
        self.assertEqual(info, {})

        # Called thrice
        self.assertEqual(orig_env.render.call_count, 3)

        obs = env.reset()
        self.assertEqual(obs, ('state', 3))

        # Called four times
        self.assertEqual(orig_env.render.call_count, 4)

        # All the calls should receive correct kwargs
        for call in orig_env.render.call_args_list:
            args, kwargs = call
            self.assertEqual(len(args), 0)
            self.assertEqual(kwargs, self.render_kwargs)
