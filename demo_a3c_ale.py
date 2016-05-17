import argparse
import os

import numpy as np
import chainer
from chainer import serializers

import ale
import random_seed
from dqn_phi import dqn_phi
from a3c_ale import A3CFF
from a3c_ale import A3CLSTM


def eval_performance(rom, model, deterministic=False, use_sdl=False,
                     record_screen_dir=None):
    env = ale.ALE(rom, treat_life_lost_as_terminal=False, use_sdl=use_sdl,
                  record_screen_dir=record_screen_dir)
    model.reset_state()
    test_r = 0
    while not env.is_terminal:
        s = chainer.Variable(np.expand_dims(dqn_phi(env.state), 0))
        pout = model.pi_and_v(s)[0]
        model.unchain_backward()
        if deterministic:
            a = pout.most_probable_actions[0]
        else:
            a = pout.action_indices[0]
        test_r += env.receive_action(a)
    return test_r


def main():

    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('rom', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-sdl', action='store_true')
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--record-screen-dir', type=str, default=None)
    parser.add_argument('--use-lstm', action='store_true')
    parser.set_defaults(use_sdl=False)
    parser.set_defaults(use_lstm=False)
    parser.set_defaults(deterministic=False)
    args = parser.parse_args()

    random_seed.set_random_seed(args.seed)

    n_actions = ale.ALE(args.rom).number_of_actions

    # Load an A3C-DQN model
    if args.use_lstm:
        model = A3CLSTM(n_actions)
    else:
        model = A3CFF(n_actions)
    serializers.load_hdf5(args.model, model)

    scores = []
    for i in range(args.n_runs):
        episode_record_dir = None
        if args.record_screen_dir is not None:
            episode_record_dir = os.path.join(args.record_screen_dir, str(i))
            os.makedirs(episode_record_dir)
        score = eval_performance(
            args.rom, model, deterministic=args.deterministic,
            use_sdl=args.use_sdl, record_screen_dir=episode_record_dir)
        print('Run {}: {}'.format(i, score))
        scores.append(score)
    print('Average: {}'.format(sum(scores) / args.n_runs))


if __name__ == '__main__':
    main()
