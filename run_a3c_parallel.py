import threading

import chainer
from chainer import optimizers

import policy
import v_function
import a3c
import xor


def use_parameter_of(a, b):
    for name in a._params:
        a[name] = b[name]


def main():
    shared_pi = policy.FCSoftmaxPolicy(2, 2, 10, 2)
    shared_v = v_function.FCVFunction(2, 10, 2)

    threads = []

    for _ in xrange(8):

        def run():

            pi = policy.FCSoftmaxPolicy(2, 2, 10, 2)
            v = v_function.FCVFunction(2, 10, 2)

            use_parameter_of(pi, shared_pi)
            use_parameter_of(v, shared_v)

            optimizer = optimizers.Adam()
            optimizer.setup(chainer.ChainList(pi, v))
            agent = a3c.A3C(pi, v, optimizer, 3, 0.9)
            env = xor.XOR()

            for i in xrange(5000):
                is_terminal = i % 2 == 1
                action = agent.act(env.state, env.reward, is_terminal)
                if is_terminal:
                    print 'r:{}', env.reward
                # print 's', env.state, 'r', env.reward, 'a', action
                env.receive_action(action)

        threads.append(threading.Thread(target=run))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
