import chainer
import numpy

import chainerrl


def test_namedpersistent():
    # This test case is adopted from
    # https://github.com/chainer/chainer/pull/6788

    l1 = chainer.Link()
    with l1.init_scope():
        l1.x = chainer.Parameter(shape=(2, 3))

    l2 = chainer.Link()
    with l2.init_scope():
        l2.x = chainer.Parameter(shape=2)
    l2.add_persistent(
        'l2_a', numpy.array([1, 2, 3], dtype=numpy.float32))

    l3 = chainer.Link()
    with l3.init_scope():
        l3.x = chainer.Parameter()
    l3.add_persistent(
        'l3_a', numpy.array([1, 2, 3], dtype=numpy.float32))

    c1 = chainer.Chain()
    with c1.init_scope():
        c1.l1 = l1
    c1.add_link('l2', l2)
    c1.add_persistent(
        'c1_a', numpy.array([1, 2, 3], dtype=numpy.float32))

    c2 = chainer.Chain()
    with c2.init_scope():
        c2.c1 = c1
        c2.l3 = l3
    c2.add_persistent(
        'c2_a', numpy.array([1, 2, 3], dtype=numpy.float32))
    namedpersistent = list(chainerrl.misc.namedpersistent(c2))
    assert (
        [(name, id(p)) for name, p in namedpersistent] ==
        [('/c2_a', id(c2.c2_a)), ('/c1/c1_a', id(c2.c1.c1_a)),
         ('/c1/l2/l2_a', id(c2.c1.l2.l2_a)), ('/l3/l3_a', id(c2.l3.l3_a))])
