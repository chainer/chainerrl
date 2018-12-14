import chainer


def batch_states(states, xp, phi):
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        xp (module): numpy or cupy
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """
    if chainer.cuda.available and xp is chainer.cuda.cupy:
        # GPU
        device = chainer.cuda.Device().id
    else:
        # CPU
        device = -1

    features = [phi(s) for s in states]
    return chainer.dataset.concat_examples(features, device=device)
