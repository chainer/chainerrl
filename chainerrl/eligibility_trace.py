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
    Q_ret = np.zeros(n+1, dtype=np.float32)
    Q_ret[n] = R
    for i in reversed(range(n)):
        tmp = rewards[i] + gamma * Q_ret[i+1]
        assert np.isscalar(tmp)
        Q_ret[i] = correction_coefs[i] * (tmp - Q[i]) + values[i]
    return Q_ret[0:n]
