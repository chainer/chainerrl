import chainer.functions as F


def quantile_loss(y, t, tau, reduce='no'):
    assert reduce == 'no', 'TODO'

    e = t - y
    return F.maximum(
        e * tau,
        e * (tau - 1.)
    )

def quantile_huber_loss_Aravkin(y, t, tau, delta=1.):
    """
    1402.4624
    """
    assert False
    e = t - y
    # if e in the interval [- delta * tau, delta * (1 - tau)]
    e ** 2 / (2 * delta)
    # else
    F.maximum(
        e * tau - delta * tau**2 / 2,
        e * (tau - 1.) - delta * (tau - 1.)**2 / 2
    )

def quantile_huber_loss_Dabney(y, t, tau, delta=1.):
    """
    1710.10044
    """

    # TODO
    assert False
