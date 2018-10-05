import math
from chainer.functions.math import ndtr
from chainer.functions.math import exponential

PROBC = 1. / (2 * math.pi) ** 0.5

def cdf(x, loc, scale):
    return ndtr.ndtr((x - loc) / scale)

def prob(x, loc, scale):
    return (PROBC / scale) * exponential.exp(
        - 0.5 * (x - loc) ** 2 / scale ** 2)

def estimate(xp, p_means, p_sigmas, n, std=3):
    start = (p_means-p_sigmas*std)
    end = (p_means+p_sigmas*std)

    interval = (end-start)/n

    mean = 0
    sigma = 0

    act_probs = xp.ones((p_means.shape[0], p_means.shape[1]), dtype=xp.float32) * 1e-5

    for i in range(n):
        alpha = start + interval*i

        def get_prob(x):
            block = xp.repeat(x, 3, axis=1).flatten()
            pdfs = prob(x.flatten(), p_means.flatten(), p_sigmas.flatten()).reshape((p_means.shape[0], p_means.shape[1]))
            #cdfs = cdf(x, p_means.flatten(), p_sigmas.flatten()).reshape((p_means.shape[0], p_means.shape[1]))
            cdfs = cdf(block, xp.tile(p_means, (1, p_means.shape[1])).flatten(),
                xp.tile(p_sigmas, (1, p_means.shape[1])).flatten()).reshape((p_means.shape[0], p_means.shape[1], p_means.shape[1]))

            a_probs = xp.zeros_like(p_means)

            for a in range(p_means.shape[1]):
                p = pdfs[:, a]
                for a2 in range(p_means.shape[1]):
                    if a2 != a:
                        p *= cdfs[:, a, a2]

                a_probs[:, a] = p.data

            return a_probs

        est = get_prob(alpha)
        act_probs += est

    act_probs2 = act_probs / act_probs.sum(axis=1)[:, None]

    return act_probs2
