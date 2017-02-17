def batch_states(states, xp, phi):
    states = [phi(s) for s in states]
    return xp.asarray(states)
