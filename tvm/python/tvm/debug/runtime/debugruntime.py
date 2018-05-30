from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import re
import threading
import logging

from ..wrappers import local_cli_wrapper as tvmdbg
from tvm import ndarray as nd

def create(obj, graph):
    """Check if an object is of the expected type.

    Args:
      obj: The object being checked.
      expected_types: (`type` or an iterable of `type`s) The expected `type`(s)
        of obj.

    Raises:
        TypeError: If obj is not an instance of expected_type.
    """
    #PRINT()
    obj.graph_json_str = graph
    graph_json = graph
    alpha = 'list_shape'
    startpos = graph_json.find(alpha) + len(alpha) + 4
    endpos = graph_json.find(']]', startpos)
    shapes_str = graph_json[startpos:(endpos + 1)]
    shape_startpos = shape_endpos = 0
    obj.ndarraylist = []
    dtype = 'float32' #TODO: dtype parse from json
    while shape_endpos < endpos - startpos:
        shape_startpos = shapes_str.find('[', shape_startpos) + 1
        shape_endpos = shapes_str.find(']', shape_startpos)
        shape_str = shapes_str[shape_startpos:shape_endpos]
        shape_list = [int(x) for x in shape_str.split(',')]
        obj.ndarraylist.append(nd.empty(shape_list, dtype))
    return tvmdbg.LocalCLIDebugWrapperSession(obj, graph)

