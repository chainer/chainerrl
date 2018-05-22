from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import subprocess

import chainer.computational_graph
import chainerrl


def collect_variables(obj):
    """Collect Variable objects inside a given object.

    Args:
        obj (object): Object to collect Variable objects from.
    Returns:
        List of Variable objects.
    """
    variables = []
    if isinstance(obj, chainer.Variable):
        return [obj]
    elif isinstance(obj, chainerrl.action_value.ActionValue):
        return list(obj.params)
    elif isinstance(obj, chainerrl.distribution.Distribution):
        return list(obj.params)
    elif isinstance(obj, (list, tuple)):
        variables = []
        for child in obj:
            variables.extend(collect_variables(child))
        return variables


def is_graphviz_available():
    return chainerrl.misc.is_return_code_zero(['dot', '-V'])


def draw_computational_graph(outputs, filepath):
    """Draw a computational graph and write to a given file.

    Args:
        outputs (object): Output(s) of the computational graph. It must be
            a Variable, an ActionValue, a Distribution or a list of them.
        filepath (str): Filepath to write a graph without file extention.
            A DOT file will be saved with ".gv" extension added.
            If Graphviz's dot command is available, a PNG file will also be
            saved with ".png" extension added.
    """
    variables = collect_variables(outputs)
    g = chainer.computational_graph.build_computational_graph(variables)
    gv_filepath = filepath + '.gv'
    with open(gv_filepath, 'w') as f:
        # future.builtins.str is required to make sure the content is unicode
        # in both py2 and py3
        f.write(str(g.dump()))
    if is_graphviz_available():
        png_filepath = filepath + '.png'
        subprocess.check_call(
            ['dot', '-Tpng', gv_filepath, '-o', png_filepath])
