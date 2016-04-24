import unittest

import replay_buffer


class TestReplayBuffer(unittest.TestCase):

    def test_append_and_sample(self):
        rbuf = replay_buffer.ReplayBuffer(100)

        self.assertEqual(len(rbuf), 0)

        # Add one and sample one
        trans1 = dict(state=0, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans1)
        self.assertEqual(len(rbuf), 1)
        s1 = rbuf.sample(1)
        self.assertEqual(len(s1), 1)
        self.assertEqual(s1[0], trans1)

        # Add two and sample two, which must be unique
        trans2 = dict(state=1, action=1, reward=2, next_state=3,
                      next_action=4, is_state_terminal=True)
        rbuf.append(**trans2)
        self.assertEqual(len(rbuf), 2)
        s2 = rbuf.sample(2)
        self.assertEqual(len(s2), 2)
        if s2[0]['state'] == 0:
            self.assertEqual(s2[0], trans1)
            self.assertEqual(s2[1], trans2)
        else:
            self.assertEqual(s2[0], trans2)
            self.assertEqual(s2[1], trans1)
