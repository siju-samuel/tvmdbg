"""Graph debug runtime executes TVM debug packed functions."""

import os
import json
import tempfile

from tvm import ndarray as nd
from tvm._ffi.base import string_types
from tvm.contrib import graph_runtime
from tvm._ffi.function import get_global_func
from tvm.contrib.debugger.curses.wrappers import ui_wrapper as tvmdbg
from tvm.contrib.debugger.curses.util import common
from . import debug_result

#Todo: User will have an option to select the frontend when debug is enabling.
#String used for frontend seletion.
#Currently only FRONTEND_CURSES is supported.
#FRONTEND_CURSES : Select the CLI cursus ui framework for UX
#FRONTEND_TESNORBOARD : Select tensorbox as the UX
FRONTEND_CURSES = 'cli'
FRONTEND_TESNORBOARD = 'tensorboard'
_DUMP_ROOT_PREFIX = "tvmdbg_"

def create(graph_json_str, libmod, ctx, frontend=FRONTEND_CURSES):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.

    libmod : tvm.Module
        The module of the corresponding function.

    ctx : TVMContext
        The context to deploy the module, can be local or remote.

    frontend : str
        To select which ui user needs, by default its curses ui.

    Returns
    -------
    graph_module : GraphModuleDebug
        Debug Runtime graph module that can be used to execute the graph.
    """
    if not isinstance(graph_json_str, string_types):
        try:
            graph_json_str = graph_json_str._tvm_graph_json()
        except AttributeError:
            raise ValueError("Type %s is not supported" % type(graph_json_str))
    device_type = ctx.device_type
    device_id = ctx.device_id
    try:
        fcreate = get_global_func("tvm.graph_runtime_debug.create")
    except ValueError:
        raise ValueError("Please set '(USE_GRAPH_RUNTIME_DEBUG ON)' in " \
                         "config.cmake and rebuild TVM to enable debug mode")
    func_obj = fcreate(graph_json_str, libmod, device_type, device_id)
    return GraphModuleDebug(func_obj, ctx, graph_json_str, frontend)


class GraphModuleDebug(graph_runtime.GraphModule):
    """Graph debug runtime module.

    This is a debug wrapper over the TVM runtime.
    Runtime interfaces are wrapped with debug functionalities.
    Manage the debug framework to format the debug data and
    trigger the user interfaces.

    Parameters
    ----------
    module : Module
        The interal tvm module that holds the actual graph functions.

    ctx : TVMContext
        The context this module is under.

    graph_json_str : str or graph class
        The graph to be deployed in json format output by nnvm graph.
        The graph can only contain one operator(tvm_op) that
        points to the name of PackedFunc in the libmod.

    frontend : str
        To select which ui user needs, curses, tensorboard, etc
    """
    def __init__(self, module, ctx, graph_json_str, frontend):
        self._set_debug_buffer = module["set_debug_buffer"]
        self._debug_run = module["debug_run"]
        graph_runtime.GraphModule.__init__(self, module, ctx)
        self._prepare_data_and_ui(graph_json_str, ctx, frontend)

    def _format_context(self, ctx):
        return str(ctx).upper().replace("(", ":").replace(")", "")

    def _prepare_data_and_ui(self, graph_json, ctx, frontend):
        """Create the framework for debug data dumpling and initialize the frontend

        Parameters
        ----------
        graph_json : str or graph class
            The graph to be deployed in json format output by nnvm graph.
            The graph can only contain one operator(tvm_op) that
            points to the name of PackedFunc in the libmod.
            value : the input value.
               The input key

        ctx : TVMContext
            The context this module is under.

        frontend: str
            'curses'- involve curses based CLI frontend
            'tensorboard'- make data format for tensorbard frontend.
        """
        nodes_list, dltype_list, shapes_list = self._parse_graph(graph_json)
        self._update_graph_json(nodes_list, dltype_list, shapes_list)

        #format the context
        ctx = self._format_context(ctx)

        self.ui_obj = self._create_debug_ui(graph_json, ctx, frontend)

        # prepare the debug out buffer list
        self.dbg_buff_list = self._make_debug_buffer_list(shapes_list, dltype_list)

        # init the debug dumping environment
        self.debug_datum = debug_result.DebugResult(nodes_list, self.dbg_buff_list, self._dump_path)

        # dump the json information
        self.debug_datum.dump_graph_json(graph_json)

    def _parse_graph(self, graph_json):
        """Parse and extract the NNVM graph.

        Parameters
        ----------
        graph : str or graph class
           The graph to be deployed in json format output by nnvm graph.

        Returns
        -------
        nodes_list : list
            List of all the nodes presented in the graph

        shapes_list : list
            List of shape of each nodes presented in the graph

        dltype_list : list
            List of data type of each nodes presented in the graph
        """
        json_obj = json.loads(graph_json)
        nodes_list = json_obj['nodes']
        dltype_list = json_obj['attrs']['dltype']
        shapes_list = json_obj['attrs']['shape']
        return nodes_list, dltype_list, shapes_list

    def _update_graph_json(self, nodes_list, dltype_list, shapes_list):
        """update the nodes_list with name, shape and data type,
        for temporarily storing the output.

        Parameters
        ----------
        nodes_list : List
            List of nodes in the graph

        dltype_list : List
            List of datatypes of each node

        shapes_list : List
            List of shape of each node

        Returns
        -------
        None
        """

        nodes_len = len(nodes_list)
        for i in range(nodes_len):
            node = nodes_list[i]
            input_list = []
            for input_node in node['inputs']:
                input_list.append(nodes_list[input_node[0]]['name'])
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

    def _get_debug_buffer_count(self):
        return len(self.dbg_buff_list)

    def _get_debug_buffer(self, eid):
        return self.dbg_buff_list[eid]

    def set_debug_buffer(self):
        """Set output buffer allocated for each node to copy the node's
        output after Run completed.

        This function will get called before run performs.
        GraphRuntime copy the execution out to the allocated memory for each nodes.

        Parameters
        ----------
          None
        """
        for eid in range(self._get_debug_buffer_count()):
            self._set_debug_buffer(self._get_debug_buffer(eid))

    def _create_debug_ui(self, graph_json, ctx, frontend):
        """Create UI wrapper framework to handle multiple UI frontends for tvmdbg

        Parameters
        ----------
        graph_json : json format
            json formatted NNVM graph contain list of each node's name, shape and type.

        nodes_list : list
            List of all the nodes presented in the graph

        ctx : TVMContext
            The context this module is under.

        frontend : str
            'curses'- involve curses based CLI frontend
            'tensorboard'- make data format for tensorbard frontend.

        Returns
        -------
        ui_wrapper : DebugGraphUIWrapper object
            UI warpper manage tvmdbg frontend.
        """
        #make the dump folder
        dump_root = tempfile.mktemp(prefix=_DUMP_ROOT_PREFIX)

        ui_wrapper = DebugGraphUIWrapper(dump_root, graph_json, ctx, frontend)

        #updates the dumping directories
        self._dump_root = dump_root
        self._dump_path = ui_wrapper.get_dump_path(ctx)
        return ui_wrapper

    def _make_debug_buffer_list(self, shapes_list, dltype_list):
        """Allocate output buffer for each node to copy the node's
        output after Run completed.

        Parameters
        ----------
        shapes_list : list
            List of shape of each nodes presented in the graph.

        dltype_list : list
            List of data type of each nodes presented in the graph.

        Returns
        -------
        dbg_out_buffer_list : list
            Allocated empty buffer
        """
        dbg_out_buffer_list = []
        for i in range(len(shapes_list[1])):
            dbg_out_buffer_list.append(nd.empty(shapes_list[1][i], dltype_list[1][i]))
        return dbg_out_buffer_list

    def _debug_run_op_exec(self, index=None):
        """Execute the node spcified with index will be executed.

        Time consumed for each execuion will be set as debug output.

        Parameters
        ----------
        index : int
            Node index to be executed now

        Returns
        -------
          none
        """
        if index:
            time_stamp = self._debug_run(index)
            self.debug_datum.set_time(time_stamp)
            return

        nodes_count = len(self.debug_datum.get_nodes_list())
        for i in range(nodes_count):
            time_stamp = self._debug_run(i)
            self.debug_datum.set_time(time_stamp)

    def _debug_cli_run(self):
        """Invoke cli and when user execute select any operation,
        'get_run_start_resp' return the user input.

        Based on the user input, different type of Run will perform.
        'set_debug_buffer' will set the empty buffer for setting node outputs.
        Once the execution compled, output will be in the dump path and CLI will
        be notified as run ends.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        while True:
            cli_command = self.ui_obj.get_run_command()
            run_start_resp = cli_command.get_run_start_resp()
            retvals = True
            if run_start_resp.action == common.CLIRunStartAction.DEBUG_RUN:
                self.set_debug_buffer()
                retvals = self._debug_run_op_exec()
                self.debug_datum.dump_output_tensor()
                self.ui_obj.run_end(cli_command, retvals)
            elif run_start_resp.action == common.CLIRunStartAction.NON_DEBUG_RUN:
                retvals = super(GraphModuleDebug, self).run()
                self.ui_obj.run_end(cli_command, retvals)
            else:
                break
        self.ui_obj.exit()

    def run(self, **input_dict):
        """Run forward execution of the graph with debug

        Parameters
        ----------
        input_dict : dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._debug_cli_run()

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Along with the value setting to runtime, the same will be notified to
        UI frontend as well.

        Parameters
        ----------

        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additonal arguments

        Returns
        -------
        none
        """
        super(GraphModuleDebug, self).set_input(key, value, **params)

        if key:
            self.ui_obj.set_input(key, value)


class DebugGraphUIWrapper(object):
    """UI Wrapper module for debug runtime

    This is a thin wrapper of the debug for TVM runtime.
    Create the UI fronted framework for tvmdbg, includes
    initialization and interfacing.


    Parameters
    ----------
    dump_root : str
        The dump folder for graph and tensors.

    graph_json : json format
        json formatted NNVM graph contain list of each node's name, shape and type.

    ctx : TVMContext
       The context this module is under.

    frontend : str
        'curses'- involve curses based CLI frontend
        'tensorboard'- make data format for tensorbard frontend.
    """
    def __init__(self, dump_root, graph_json, ctx, frontend):
        """Init the DebugGraphUIWrapper"""
        if frontend == FRONTEND_CURSES:
            self.curses_obj = tvmdbg.LocalCLIDebugWrapperModule(self,
                                                                graph_json,
                                                                ctx=ctx,
                                                                dump_root=dump_root)

    def get_run_command(self):
        """Invoke run from curses ui"""
        return self.curses_obj.get_run_command()

    def run_end(self, run_cli_session, retvals):
        """Invoke run end from curses ui"""
        self.curses_obj.run_end(run_cli_session, retvals)

    def set_input(self, key, value):
        """Set inputs to curses ui"""
        self.curses_obj.set_input(key.replace("/", "_"), value)

    def exit(self):
        """Exits the curses ui"""
        self.curses_obj.exit()

    def set_output_nodes(self, heads_list):
        """Dump the heads to a list

        Parameters
        ----------

        cli_obj : obj
            The CLI object

        heads_list : List
            The list of outputs from the json node

        """
        for output in heads_list:
            self.curses_obj.set_ouputs(self._nodes_list[output[0]]['name'])

    def _ensure_dir(self, file_path):
        """Create a directory if not exists

        Parameters
        ----------

        file_path : str
            File path to create

        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_dump_path(self, ctx):
        """Dump json formatted graph.

        Parameters
        ----------
        ctx : TVMContext
            The context this module is under.

        Returns
        -------
        path : str
            Directory path where the graph and node outputs will be stored.
        """
        # save to file
        folder_name = "/_tvmdbg_device_,device_"
        folder_name = folder_name + ctx.replace(":", "_") + "/"
        self.curses_obj.dump_folder(folder_name)
        path = self.curses_obj._dump_root + folder_name
        self._ensure_dir(path)
        return path
