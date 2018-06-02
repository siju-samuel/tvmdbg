"""Debug runtime functions."""
import json
from tvm import ndarray as nd
from ..wrappers import local_cli_wrapper as tvmdbg

def _dump_json(nodes_list, dltype_list, shapes_list):
    """Create a debug runtime environment and start the CLI

    Args:
      nodes_list: List of the nodes in the graph and their details
      dltype_list: List of datatypes of each node
      shapes_list: List of shape of each node
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

def create(obj, graph):
    """Create a debug runtime environment and start the CLI

    Args:
      obj: The object being used to store the graph runtime.
      graph: nnvm graph in json format
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