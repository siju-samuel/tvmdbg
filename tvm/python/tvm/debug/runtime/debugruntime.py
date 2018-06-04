"""Debug runtime functions."""
import json
import os
import numpy as np
from tvm import ndarray as nd
from ..wrappers import local_cli_wrapper as tvmdbg

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def _dump_json(cli_obj, nodes_list, dltype_list, shapes_list):
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

    path = cli_obj._dump_root
    folder_name = "/_tvmdbg_device_,job_localhost,replica_0,task_0,device_CPU_0/"
    cli_obj._dump_folder = folder_name
    ensure_dir(path+folder_name)
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
        else :
            #node['op'] = node['name']
            node['op'] = node['attrs']['func_name']
        node['attrs'].update({"T":dltype})
        node['shape'] = shapes_list[1][i]
        new_graph['nodes'].append(node)
    #save to file
    graph_dump_file_path = '_tvmdbg_graph_dump.json'

    with open((path + folder_name + graph_dump_file_path), 'w') as outfile:
        json.dump(new_graph, outfile, indent=2, sort_keys=False)

def _dump_input(cli_obj, file_name, key, value):
    np.save(str(cli_obj._dump_root + cli_obj._dump_folder + key + file_name), value.asnumpy())

def dump_output(cli_obj, ndarraylist):
    for i in range (len(ndarraylist)):
        ndbuffer = ndarraylist[i]
        if cli_obj._nodes_list[i]['op'] != 'null':
            #        `node_name`_`output_slot`_`debug_op`_`timestamp`
            key = cli_obj._nodes_list[i]['name'] + "_0_DebugIdentity_000000" + str(i) + ".npy"
            file_name = str(cli_obj._dump_root + cli_obj._dump_folder + key)
            np.save(file_name, ndbuffer.asnumpy())
            os.rename(file_name, file_name.rpartition('.')[0])

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
        #_dump_input(cli_obj, '_value_dump', key, value)
        cli_obj.set_input(key, value);

    for k, v in params.items():
        #_dump_input(cli_obj, '_value.json', k, v)
        cli_obj.set_input(k, v);

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
    print("dump directory", cli_obj._dump_root)
    json_obj=json.loads(graph)
    cli_obj._nodes_list =json_obj['nodes']
    dltype_list = json_obj['attrs']['dltype']
    shapes_list = json_obj['attrs']['shape']
    #dump the json information
    _dump_json(cli_obj, cli_obj._nodes_list, dltype_list, shapes_list)
    #prepare the out shape
    obj.ndarraylist = []
    for i in range (len(shapes_list[1])):
        shape = shapes_list[1][i]
        obj.ndarraylist.append(nd.empty(shapes_list[1][i], dltype_list[1][i]))
    return cli_obj