import chainer
from chainer import optimizers

import policy
import v_function
import a3c
import xor


def main():
    pi = policy.FCSoftmaxPolicy(2, 2, 10, 2)
    v = v_function.FCVFunction(2, 10, 2)
    optimizer = optimizers.Adam()
    optimizer.setup(chainer.ChainList(pi, v))

    agent = a3c.A3C(pi, v, optimizer, 3, 0.9)
    env = xor.XOR()

    for i in xrange(10000):
        action = agent.act(env.state, env.reward, i % 5 == 1)
        print 's', env.state, 'r', env.reward, 'a', action
        env.receive_action(action)

if __name__ == '__main__':
    main()
