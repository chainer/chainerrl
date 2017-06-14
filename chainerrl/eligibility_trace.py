import numpy as np


def lambda_return(lambd, rewards, next_values, gamma, R=0.0):
    n = rewards.size
    ret = np.zeros(n)
    for i in reversed(range(n)):
        # R = lambd * (rewards[i] + gamma * R) \
        #     + (1 - lambd) * (rewards[i] + gamma * next_values[i])
        R = lambd * R + (1 - lambd) * next_values[i]
        R = rewards[i] + gamma * R
        ret[i] = R
    return ret


def retrace(Q, rewards, values, gamma, likelihood_ratio, R=0.0, lambd=1.0):
    return general_trace(
        Q, rewards, values, gamma,
        np.clip(likelihood_ratio, None, lambd),
        R=R)


def general_trace(Q, rewards, values, gamma, correction_coefs, R=0.0):
    n = Q.size
    assert rewards.shape == (n,)
    assert values.shape == (n,)
    assert correction_coefs.shape == (n,)
    assert np.isscalar(R)
    Q_ret = np.zeros(n, dtype=np.float32)
    for i in reversed(range(n)):
        tmp = rewards[i] + gamma * R
        assert np.isscalar(tmp)
        R = Q_ret[i] = correction_coefs[i] * (tmp - Q[i]) + values[i]
    return Q_ret
