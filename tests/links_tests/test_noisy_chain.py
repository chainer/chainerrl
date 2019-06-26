import unittest

import chainer
import chainer.testing
import numpy

import chainerrl
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

    def test_sequence(self):
        model = chainerrl.links.Sequence(
            chainer.links.Linear(3, 4),
            chainer.functions.relu,
            chainer.links.Linear(4, 5),
        )
        self.assertEqual(
            names_of_links(model),
            {'/0', '/1'}
        )
        self.assertIs(model.layers[1], chainer.functions.relu)
        to_factorized_noisy(model)
        self.assertEqual(
            names_of_links(model),
            {
                '/0', '/0/mu', '/0/sigma',
                '/1', '/1/mu', '/1/sigma',
            })
        self.assertIs(model.layers[1], chainer.functions.relu)
        model.cleargrads()

        # assert new parameters are used
        y = model(numpy.ones((2, 3), numpy.float32))
        chainer.functions.sum(y).backward()
        for p in model.params():
            self.assertIsNotNone(p.grad)

    @chainer.testing.with_requires('chainer>=5')
    def test_sequential(self):
        model = chainer.Sequential(
            chainer.links.Linear(3),
            chainer.functions.relu,
            chainer.links.Linear(4),
        )
        self.assertEqual(
            names_of_links(model),
            {'/0', '/1'}
        )
        self.assertIs(model._layers[1], chainer.functions.relu)
        to_factorized_noisy(model)
        self.assertEqual(
            names_of_links(model),
            {
                '/0', '/0/mu', '/0/sigma',
                '/1', '/1/mu', '/1/sigma',
            })
        self.assertIs(model._layers[1], chainer.functions.relu)
        model(numpy.ones((2, 3), numpy.float32))

        # assert new parameters are used
        y = model(numpy.ones((2, 3), numpy.float32))
        chainer.functions.sum(y).backward()
        for p in model.params():
            self.assertIsNotNone(p.grad)
