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

    encoded_states = [phi(s) for s in states]
    if any(chainer.cuda.get_array_module(s) is chainer.cuda.cupy
            for s in encoded_states):
        # xp must be cupy when some of observations is in gpu
        assert xp == chainer.cuda.cupy
        return xp.stack([xp.asarray(s) for s in encoded_states])
    elif xp == numpy:
        # All elements should be numpy.ndarray
        return xp.stack(encoded_states)
    else:
        return xp.asarray(encoded_states)
