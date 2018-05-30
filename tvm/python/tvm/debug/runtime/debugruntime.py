"""Debug runtime functions."""

from tvm import ndarray as nd
from ..wrappers import local_cli_wrapper as tvmdbg

def create(obj, graph):
    """Create a debug runtime environment and start the CLI

    Args:
      obj: The object being used to store the graph runtime.
      graph: nnvm graph in json format
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
