"""Debug runtime functions."""
import json
import numpy as np
from tvm import ndarray as nd
from ..wrappers import local_cli_wrapper as tvmdbg

def _dump_json(nodes_list, dltype_list, shapes_list):
    """Create a debug runtime environment and start the CLI

    Parameters
    ----------
    nodes_list: List
        List of the nodes in the graph and their details
    dltype_list: List
        List of datatypes of each node
    shapes_list: List
        List of shape of each node
    """
    new_graph = {}
    new_graph['nodes'] = []
    for  i in range (len(nodes_list)):
        node = nodes_list[i]
        input_list = []
        for input in (node['inputs']):
            input_list.append(nodes_list[input[0]]['name'])
        node['inputs'] = input_list
        if not len(node['inputs']):
            del node['inputs']
        dltype = str("type: " +  dltype_list[1][i])
        if 'attrs' not in node:
            node['attrs'] = {}
        node['attrs'].update({"T":dltype})
        node['shape'] = shapes_list[1][i]
        new_graph['nodes'].append(node)
    #save to file
    graph_dump_file_path = 'graph_dump.json'

    with open(graph_dump_file_path, 'w') as outfile:
        json.dump(new_graph, outfile, indent=2, sort_keys=False)

def _dump_input(file_name, key, value):
    np.save(str(key + file_name), value.asnumpy())

def set_input(cli_obj, key=None, value=None, **params):
    """Set inputs to the module via kwargs

    Parameters
    ----------
    cli_obj: obj
        The CLI object

    key : int or str
       The input key

    value : the input value.
       The input key

    params : dict of str to NDArray
       Additonal arguments
    """
    if key:
        _dump_input('_value_dump', key, value)

    for k, v in params.items():
        _dump_input('_value.json', k, v)

def create(obj, graph):
    """Create a debug runtime environment and start the CLI

    Parameters
    ----------
    obj: Object
        The object being used to store the graph runtime.
    graph: str
        nnvm graph in json format
    """

    cli_obj = tvmdbg.LocalCLIDebugWrapperSession(obj, graph)
    json_obj=json.loads(graph)
    nodes_list =json_obj['nodes']
    dltype_list = json_obj['attrs']['dltype']
    shapes_list = json_obj['attrs']['shape']
    #dump the json information
    _dump_json(nodes_list, dltype_list, shapes_list)
    #prepare the out shape
    obj.ndarraylist = []
    for i in range (len(shapes_list[1])):
        shape = shapes_list[1][i]
        obj.ndarraylist.append(nd.empty(shapes_list[1][i], dltype_list[1][i]))
    return cli_obj