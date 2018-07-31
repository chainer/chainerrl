import unittest

import chainer

from chainerrl.links import to_factorized_noisy


def names_of_links(link):
    return set([name for name, _ in link.namedlinks(skipself=True)])


class TestToFactorizedNoisy(unittest.TestCase):
    def test_chainlist(self):
        ch = chainer.ChainList(
            chainer.links.Linear(3, 4),
            chainer.links.Linear(5),
            chainer.links.PReLU(),
        )
        self.assertEqual(
            names_of_links(ch),
            {'/0', '/1', '/2'})

        to_factorized_noisy(ch)
        self.assertEqual(
            names_of_links(ch),
            {
                '/0', '/0/mu', '/0/sigma',
                '/1', '/1/mu', '/1/sigma', '/2'})

    def test_chain(self):
        ch = chainer.Chain()
        with ch.init_scope():
            ch.l1 = chainer.links.Linear(3, 4)
            ch.l2 = chainer.links.Linear(5)
            ch.l3 = chainer.links.PReLU()
        self.assertEqual(
            names_of_links(ch),
            {'/l1', '/l2', '/l3'})

        to_factorized_noisy(ch)
        self.assertEqual(
            names_of_links(ch),
            {
                '/l1', '/l1/mu', '/l1/sigma',
                '/l2', '/l2/mu', '/l2/sigma', '/l3'})
