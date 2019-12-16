import logging
import os

import chainer


def collect_demonstrations(agent,
                           env,
                           steps,
                           episodes,
                           outdir,
                           max_episode_len=None,
                           logger=None):
    """Collects demonstrations from an agent and writes them to a file.

    Args:
        agent: Agent from which demonstrations are collected.
        env: Environment in which the agent produces demonstrations.
        steps (int): Number of total time steps to collect demonstrations for.
        episodes (int): Number of total episodes to collect demonstrations for.
        outdir (str): Path to the directory to output demonstrations.
        max_episode_len (int): Maximum episode length.
        logger (logging.Logger): Logger used in this function.
    """
    assert (steps is None) != (episodes is None)
    logger = logger or logging.getLogger(__name__)

    with chainer.datasets.open_pickle_dataset_writer(
            os.path.join(outdir, "demos.pickle")) as dataset:
        # o_0, r_0
        terminate = False  # True if we should stop collecting demos
        timestep = 0  # number of timesteps of demos collected
        episode_num = 0  # number of episodes of demos collected
        episode_len = 0  # length of most recent episode
        reset = True  # whether to reset environment
        episode_r = 0  # Episode reward
        while not terminate:
            if reset:
                if episode_num > 0:
                    logger.info('demonstration episode %s length:%s R:%s',
                                episode_num, episode_len, episode_r)
                obs = env.reset()
                done = False
                r = 0
                episode_r = 0
                episode_len = 0
                info = {}
            # a_t
            a = agent.act(obs)
            # o_{t+1}, r_{t+1}
            new_obs, r, done, info = env.step(a)
            # o_t, a_t, r__{t+1}, o_{t+1}
            dataset.write((obs, a, r, new_obs, done, info))
            obs = new_obs
            reset = (done or episode_len == max_episode_len
                     or info.get('needs_reset', False))
            timestep += 1
            episode_len += 1
            episode_r += r
            episode_num = episode_num + 1 if reset else episode_num
            if steps is None:
                terminate = episode_num >= episodes
            else:
                terminate = timestep >= steps
            if reset or terminate:
                agent.stop_episode()
                if terminate:
                    # log final episode
                    logger.info('demonstration episode %s length:%s R:%s',
                                episode_num, episode_len, episode_r)
