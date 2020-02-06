"""This file is a fork from ChainerCV, an MIT-licensed project,
https://github.com/chainer/chainercv/blob/master/chainercv/utils/download.py
"""

import filelock
import hashlib
import os
import shutil
import tempfile
import time
import sys
import zipfile
from six.moves.urllib import request

from chainer.dataset.download import get_dataset_directory
from chainer.dataset.download import get_dataset_root


MODELS = {
    "DQN": ["best", "final"],
    "IQN": ["best", "final"],
    "Rainbow": ["best", "final"],
    "A3C": ["final"],
    "DDPG": ["best", "final"],
    "TRPO": ["best", "final"],
    "PPO": ["final"],
    "TD3": ["best", "final"],
    "SAC": ["best", "final"]
}

download_url = "https://chainer-assets.preferred.jp/chainerrl/"


def _reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        print('  %   Total    Recv       Speed  Time left')
        return
    duration = time.time() - start_time
    progress_size = count * block_size
    try:
        speed = progress_size / duration
    except ZeroDivisionError:
        speed = float('inf')
    percent = progress_size / total_size * 100
    eta = int((total_size - progress_size) / speed)
    sys.stdout.write(
        '\r{:3.0f} {:4.0f}MiB {:4.0f}MiB {:6.0f}KiB/s {:4d}:{:02d}:{:02d}'
        .format(
            percent, total_size / (1 << 20), progress_size / (1 << 20),
            speed / (1 << 10), eta // 60 // 60, (eta // 60) % 60, eta % 60))
    sys.stdout.flush()


def cached_download(url):
    """Downloads a file and caches it.

    This is different from the original
    :func:`~chainer.dataset.cached_download` in that the download
    progress is reported.
    It downloads a file from the URL if there is no corresponding cache.
    If there is already a cache for the given URL, it just returns the
    path to the cache without downloading the same file.
    Args:
        url (string): URL to download from.
    Returns:
        string: Path to the downloaded file.
    """
    cache_root = os.path.join(get_dataset_root(), '_dl_cache')
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.exists(cache_root):
            raise
    lock_path = os.path.join(cache_root, '_dl_lock')
    urlhash = hashlib.md5(url.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_root, urlhash)

    with filelock.FileLock(lock_path):
        if os.path.exists(cache_path):
            return cache_path

    temp_root = tempfile.mkdtemp(dir=cache_root)
    try:
        temp_path = os.path.join(temp_root, 'dl')
        print('Downloading ...')
        print('From: {:s}'.format(url))
        print('To: {:s}'.format(cache_path))
        request.urlretrieve(url, temp_path, _reporthook)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, cache_path)
    finally:
        shutil.rmtree(temp_root)

    return cache_path


def download_and_store_model(alg, url, env, model_type):
    """Downloads a model file and puts it under model directory.

    It downloads a file from the URL and puts it under model directory.
    If there is already a file at the destination path,
    it just returns the path without downloading the same file.
    Args:
        alg (string): String representation of algorithm used in MODELS dict.
        url (string): URL to download from.
        env (string): Environment in which pretrained model was trained.
        model_type (string): Either `best` or `final`.
    Returns:
        string: Path to the downloaded file.
        bool: whether the model was alredy cached.
    """
    with filelock.FileLock(os.path.join(
            get_dataset_directory(os.path.join('pfnet', 'chainerrl', '.lock')),
            'models.lock')):
        root = get_dataset_directory(
            os.path.join('pfnet', 'chainerrl', 'models', alg, env))
        url_basepath = os.path.join(url, alg, env)
        file = model_type + ".zip"
        path = os.path.join(root, file)
        is_cached = os.path.exists(path)
        if not is_cached:
            cache_path = cached_download(os.path.join(url_basepath,
                                                      file))
            os.rename(cache_path, path)
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(root)
        return os.path.join(root, model_type), is_cached


def download_model(alg, env, model_type="best"):
    """Downloads and returns pretrained model.

    Args:
        alg (string): URL to download from.
        env (string): Gym Environment name.
        model_type (string): Either `best` or `final`.
    Returns:
        str: Path to the downloaded file.
        bool: whether the model was already cached.
    """
    assert alg in MODELS, \
        "No pretrained models for " + alg + "."
    assert model_type in MODELS[alg], \
        "Model type \"" + model_type + "\" is not supported."
    env = env.replace("NoFrameskip-v4", "")
    model_path, is_cached = download_and_store_model(alg,
                                                     download_url,
                                                     env, model_type)
    return model_path, is_cached
