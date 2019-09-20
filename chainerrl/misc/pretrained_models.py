"""This file is a fork from ChainerCV, an MIT-licensed project,
https://github.com/chainer/chainercv/blob/master/chainercv/utils/download.py
"""


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
from distutils.util import strtobool
import filelock
import hashlib
import os
import shutil
import tempfile
import time
import sys
from six.moves.urllib import request

from chainer.dataset.download import get_dataset_directory
from chainer.dataset.download import get_dataset_root
from chainer import links as L
from chainer import optimizers

import chainerrl
from chainerrl import misc

from pdb import set_trace


PRETRAINED_MODELS = {
    "DQN": ["model.npz", "target_model.npz",
            "optimizer.npz"],
    "IQN": ["model.npz", "target_model.npz",
            "optimizer.npz"],
    "Rainbow": ["model.npz", "target_model.npz",
            "optimizer.npz"],
    "A3C": ["model.npz"  "optimizer.npz"]
}

MODEL_TYPES = {
    "DQN": {"best": "best",
            "final": "5000000_finish"},
    "IQN": {"best": "best",
            "final": "5000000_finish"},
    "Rainbow": {"best": "best",
                "final": "5000000_finish"},
    "A3C": {"final": "8000000_finish"},
}

url = "https://chainer-assets.preferred.jp/chainerrl/"


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
    progress is reported. Note that this progress report can be disabled
    by setting the environment variable `CHAINERCV_DOWNLOAD_REPORT` to `'OFF'`.
    It downloads a file from the URL if there is no corresponding cache. After
    the download, this function stores a cache to the directory under the
    dataset root (see :func:`set_dataset_root`). If there is already a cache
    for the given URL, it just returns the path to the cache without
    downloading the same file.
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
        if strtobool(os.getenv('CHAINERRL_DOWNLOAD_REPORT', 'ON')):
            print('Downloading ...')
            print('From: {:s}'.format(url))
            print('To: {:s}'.format(cache_path))
            request.urlretrieve(url, temp_path, _reporthook)
        else:
            request.urlretrieve(url, temp_path)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, cache_path)
    finally:
        shutil.rmtree(temp_root)

    return cache_path

def download_and_store_model(alg, url, env, model_type):
    """Downloads a model file and puts it under model directory.
    It downloads a file from the URL and puts it under model directory.
    For example, if :obj:`url` is `http://example.com/subdir/model.npz`,
    the pretrained weights file will be saved to
    `$CHAINER_DATASET_ROOT/pfnet/chainercv/models/model.npz`.
    If there is already a file at the destination path,
    it just returns the path without downloading the same file.
    Args:
        url (string): URL to download from.
    Returns:
        string: Path to the downloaded file.
    """

    with filelock.FileLock(os.path.join(
            get_dataset_directory(os.path.join('pfnet', 'chainerrl', '.lock')),
            'models.lock')):
        model_dir = MODEL_TYPES[alg][model_type]
        root = get_dataset_directory(
            os.path.join('pfnet', 'chainerrl', 'models', alg, env, model_dir))
        url_basepath = os.path.join(url, alg, env,model_dir)
        url_paths = []
        for file in PRETRAINED_MODELS[alg]:
            # url_paths.append(os.path.join(url_basepath,
            #                     file))
            path = os.path.join(root, file)
            if not os.path.exists(path):
                cache_path = cached_download(os.path.join(url_basepath,
                                             file))
                os.rename(cache_path, path)
        return root


def download_model(alg, env, model_type="best"):
    assert alg in PRETRAINED_MODELS, \
        "No pretrained models for " + alg +"."
    assert model_type in MODEL_TYPES[alg], \
        "Model type \"" + model_type + "\" is not supported."
    env = env.replace("NoFrameskip-v4", "")
    model_path = download_and_store_model(alg, url, env, model_type)
    return model_path
