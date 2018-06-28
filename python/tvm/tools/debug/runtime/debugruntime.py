"""Debug runtime functions."""

import os
import json
import numpy as np
from tvm import ndarray as nd
from tvm._ffi.base import string_types
from tvm.contrib import graph_runtime
from tvm.contrib.rpc import base as rpc_base
from tvm._ffi.function import get_global_func
from tvm.tools.debug.wrappers import ui_wrapper as tvmdbg
from tvm.tools.debug.util import common

FRONTEND_CLI = 'cli'
FRONTEND_TESNORBOARD = 'tensorboard'

class GraphModuleDebugDumpDatum():
    def __init__(self, nodes_list, node_stats, dump_path, ctx):
        self._nodes_list = nodes_list
        self._dump_path = dump_path
        self._out_stats = node_stats
        self.ctx = ctx

    def dump_output(self):
        """Dump the outputs to a temporary folder

        Parameters
        ----------

        cli_obj: obj
            The CLI object

        """
        eid = 0
        for node in self._nodes_list:
            num_outputs = 1 if node['op'] == 'param' else int(node['attrs']['num_outputs'])
            for j in range(num_outputs):
                ndbuffer = self._out_stats[eid]
                eid += 1
                key = node['name'] + "_" + str(j) + "__000000" + str(ndbuffer.time_stamp) + ".npy"
                key = key.replace("/", "_")
                dump_file = str(self._dump_path + key)
                np.save(dump_file, ndbuffer.asnumpy())
                os.rename(dump_file, dump_file.rpartition('.')[0])

    def dump_graph_json(self, p_graph):
        graph_dump_file_name = '_tvmdbg_graph_dump.json'
        with open((self._dump_path + graph_dump_file_name), 'w') as outfile:
            json.dump(p_graph, outfile, indent=2, sort_keys=False)

class GraphModuleDebug(graph_runtime.GraphModule):
    def __init__(self, module, ctx, graph_json_str):
        self._set_debug_buffer = module["set_debug_buffer"]
        self._debug_run = module["debug_run"]
        graph_runtime.GraphModule.__init__(self, module, ctx)
        frontend = FRONTEND_CLI
        self.prepare_data_and_ui(graph_json_str, ctx, frontend)

    def prepare_data_and_ui(self, graph, ctx, frontend):
        nodes_list, dltype_list, shapes_list, heads_list = self._parse_graph(graph)
        # prepare the debug out buffer list
        self.dbg_buff_list = self._make_debug_buffer_list(shapes_list, dltype_list)
        p_graph = self._get_graph_json(nodes_list,
                                       dltype_list, shapes_list)
        ctx = str(ctx).upper().replace("(", ":").replace(")", "")
        self.ui = self._create_debug_ui(p_graph, nodes_list, heads_list, ctx, frontend)
        dump_path = self.ui.get_dump_path(ctx)
        self.debug_datum = GraphModuleDebugDumpDatum(nodes_list, self.dbg_buff_list, dump_path, ctx)
        # dump the json information
        self.debug_datum.dump_graph_json(p_graph)
        self.ui.set_output_nodes(heads_list)

    def _get_debug_buffer_count(self):
        return len(self.dbg_buff_list)

    def _get_debug_buffer(self, eid):
        return self.dbg_buff_list[eid]

    def set_debug_buffer(self):
        """Set the debug out buffers for each tvm nodes

        Parameters
        ----------
        None
        """
        for eid in range(self._get_debug_buffer_count()):
            self._set_debug_buffer(self._get_debug_buffer(eid))

    def _create_debug_ui(self, p_graph, nodes_list, heads_list, ctx, frontend):
        ctx = str(ctx).upper().replace("(", ":").replace(")", "")
        ui_wrapper = DebugGraphUIWrapper(p_graph, nodes_list,
                                         heads_list, ctx, frontend)
        return ui_wrapper

    def _parse_graph(self, graph):
        json_obj = json.loads(graph)
        nodes_list = json_obj['nodes']
        dltype_list = json_obj['attrs']['dltype']
        shapes_list = json_obj['attrs']['shape']
        heads_list = json_obj['heads']
        return nodes_list, dltype_list, shapes_list, heads_list

    def _get_graph_json(self, nodes_list, dltype_list, shapes_list):
        """Dump the nodes in json format to file

        Parameters
        ----------

        ctx: Str
            context in string

        cli_obj: obj
            CLI object where common information is stored

        nodes_list: List
            List of nodes in the graph

        dltype_list: List
            List of datatypes of each node

        shapes_list: List
            List of shape of each node

        """

        p_graph = {}
        p_graph['nodes'] = []
        nodes_len = len(nodes_list)
        for i in range(nodes_len):
            node = nodes_list[i]
            input_list = []
            for input_node in node['inputs']:
                input_list.append(nodes_list[input_node[0]]['name'])
            #del node['inputs']
            node['inputs'] = input_list
            dltype = str("type: " + dltype_list[1][i])
            if 'attrs' not in node:
                node['attrs'] = {}
                node['op'] = "param"
            else:
                node['op'] = node['attrs']['func_name']
            node['name'] = node['name'].replace("/", "_")
            node['attrs'].update({"T": dltype})
            node['shape'] = shapes_list[1][i]
            p_graph['nodes'].append(node)

        return p_graph

    def _make_debug_buffer_list(self, shapes_list, dltype_list):
        dbg_out_buffer_list = []
        for i in range(len(shapes_list[1])):
            dbg_out_buffer_list.append(nd.empty(shapes_list[1][i], dltype_list[1][i]))
        return dbg_out_buffer_list

    def _debug_cli_run(self):
        """Invoke cli and when user execute any command get_run_start_resp will return response"""
        cli_command = self.ui.get_run_command()
        run_start_resp = cli_command.get_run_start_resp()
        retvals = True
        if run_start_resp.action == common.CLIRunStartAction.DEBUG_RUN:
            self.set_debug_buffer()
            retvals = self._debug_run()
            self.debug_datum.dump_output()
            self.ui.run_end(cli_command, retvals)

        elif run_start_resp.action == common.CLIRunStartAction.NON_DEBUG_RUN:
            retvals = super(GraphModuleDebug, self).run()
            self.ui.run_end(cli_command, retvals)

    def run(self, **input_dict):
        self._debug_cli_run()

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Parameters
        ----------

        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additonal arguments
        """
        super(GraphModuleDebug, self).set_input(key, value, **params)

        if key:
            self.ui.set_input(key, value)


class DebugGraphUIWrapper(object):
    """Wrapper debug runtime module.

    This is a thin wrapper of the debug for TVM runtime.

    Parameters
    ----------
    nodes_list : list
        The list of all the graph nodes.

    cli_obj : Object
        The context of CLI object

    """
    def __init__(self, p_graph, nodes_list, heads_list, ctx, frontend):
        self._nodes_list = nodes_list
        if frontend == FRONTEND_CLI:
            self.ui_obj = tvmdbg.LocalCLIDebugWrapperModule(self, p_graph, ctx=ctx)
        self.set_output_nodes(heads_list)

    def get_run_command(self):
        return self.ui_obj.get_run_command()

    def run_end(self, run_cli_session, retvals):
        self.ui_obj.run_end(run_cli_session, retvals)

    def set_input(self, key, value):
        self.ui_obj.set_input(key.replace("/", "_"), value)

    def set_output_nodes(self, heads_list):
        """Dump the heads to a list

        Parameters
        ----------

        cli_obj: obj
            The CLI object

        heads_list : List
           The list of outputs from the json node

        """
        for output in heads_list:
            self.ui_obj.set_ouputs(self._nodes_list[output[0]]['name'])

    def _ensure_dir(self, file_path):
        """Create a directory if not exists

        Parameters
        ----------

        file_path: str
            File path to create

        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_dump_path(self, ctx):
        # save to file
        folder_name = "/_tvmdbg_device_,job_localhost,replica_0,task_0,device_"
        folder_name = folder_name + ctx.replace(":", "_") + "/"
        self.ui_obj.dump_folder(folder_name)
        path = self.ui_obj._dump_root + folder_name
        self._ensure_dir(path)
        return path

def create(graph_json_str, libmod, ctx):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.

    libmod : tvm.Module
        The module of the corresponding function

    ctx : TVMContext
        The context to deploy the module, can be local or remote.

    debug : bool
        To enable or disable the debugging

    Returns
    -------
    graph_module : GraphModule
        Runtime graph module that can be used to execute the graph.
    """
    if not isinstance(graph_json_str, string_types):
        try:
            graph_json_str = graph_json_str._tvm_graph_json()
        except AttributeError:
            raise ValueError("Type %s is not supported" % type(graph_json_str))
    device_type = ctx.device_type
    device_id = ctx.device_id
    if device_type >= rpc_base.RPC_SESS_MASK:
        assert libmod.type_key == "rpc"
        assert rpc_base._SessTableIndex(libmod) == ctx._rpc_sess._tbl_index
        hmod = rpc_base._ModuleHandle(libmod)
        fcreate = ctx._rpc_sess.get_function("tvm.graph_runtime.remote_create")
        device_type = device_type % rpc_base.RPC_SESS_MASK
        func_obj = fcreate(graph_json_str, hmod, device_type, device_id)
        return GraphModuleDebug(func_obj, ctx, graph_json_str)
    fcreate = get_global_func("tvm.graph_runtime.create")
    func_obj = fcreate(graph_json_str, libmod, device_type, device_id)
    return GraphModuleDebug(func_obj, ctx, graph_json_str)
