from distutils.version import StrictVersion
import pkg_resources

import chainer.functions as F


chainer_version = StrictVersion(
    pkg_resources.get_distribution("chainer").version)

if chainer_version < StrictVersion('3.0.0a1'):
    """Chainer's PR #2426 changed the behavior of matmul

    Simulate the newer behavior by functions in Chainer v2
    """
    def matmul_v3(a, b, **kwargs):
        if (a.ndim, b.ndim) == (3, 3):
            return F.batch_matmul(a, b, **kwargs)
        elif (a.ndim, b.ndim) == (2, 2):
            return F.matmul(a, b, **kwargs)
        else:
            raise Exception("unsupported shapes: {}, {}".format(
                a.shape, b.shape))
else:
    matmul_v3 = F.matmul
