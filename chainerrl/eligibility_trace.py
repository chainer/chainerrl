import numpy as np


def lambda_return(one_step_values, rewards, lambd, gamma):
    n = len(rewards)
    ret = [None] * n
    accum = np.array([])
    for i in reversed(range(n)):
        ret[i] = np.array(one_step_values[i])
        # print(one_step_values[i])
        # print(rewards[i])
        # print(accum)
        b = accum.shape[0]
        # print(b)
        # print(one_step_values[i][:b])
        # print(accum - one_step_values[i][:b])
        ret[i][:b] += lambd * (accum - one_step_values[i][:b])
        accum = rewards[i] + gamma * ret[i]
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
