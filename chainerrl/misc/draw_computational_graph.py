from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import subprocess
import tempfile

import chainer.computational_graph
import chainerrl


def collect_variables(obj):
    variables = []
    if isinstance(obj, chainer.Variable):
        return [obj]
    elif isinstance(obj, chainerrl.action_value.ActionValue):
        return [obj.greedy_actions(), obj.max()]
    elif isinstance(obj, chainerrl.distribution.Distribution):
        return obj.params
    elif isinstance(obj, (list, tuple)):
        variables = []
        for child in obj:
            variables.extend(collect_variables(child))
        return variables


def draw_computational_graph(outputs, filepath):
    """Draw a computational graph and write to a given file.

    Args:
        outputs (object): Output(s) of the computational graph. Each
            item must be Variable, ActionValue, Distribution or list of them.
        filepath (str): Filepath to write a graph.
            If it ends with ".dot", it will be in the dot format.
            If it ends with ".png", in the png format.
    """
    variables = collect_variables(outputs)
    g = chainer.computational_graph.build_computational_graph(variables)
    if filepath.lower().endswith('.dot'):
        with open(filepath, 'w') as f:
            f.write(g.dump())
    elif filepath.lower().endswith('.png'):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(g.dump())
            f.file.close()
            subprocess.check_call(['dot', '-Tpng', f.name, '-o', filepath])
    else:
        raise RuntimeError(
            'filepath should end with ".dot" or ".png", but is {}'.format(
                filepath))
