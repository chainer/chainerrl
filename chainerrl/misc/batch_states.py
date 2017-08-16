import chainer
import numpy


def batch_states(states, xp, phi):
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        xp (module): numpy or cupy
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """

    states = [phi(s) for s in states]
    if chainer.cuda.get_array_module(states[0]) == numpy:
        # xp can be numpy or cupy
        return xp.asarray(states)
    else:
        # xp must be cupy when observation is in gpu
        assert xp == chainer.cuda.cupy
        return xp.stack(states)
