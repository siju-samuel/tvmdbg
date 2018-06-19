"""Debugger Wrapper Session Consisting of a Local Curses-based CLI."""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import shutil
import sys
import tempfile

from tvm.tools.debug.cli import analyzer_cli
from tvm.tools.debug.cli import cli_shared
from tvm.tools.debug.cli import command_parser
from tvm.tools.debug.cli import debugger_cli_common
from tvm.tools.debug.cli import ui_factory
from tvm.tools.debug.util import common
from tvm.tools.debug.util import debug_data
from tvm.tools.debug.wrappers import framework

_DUMP_ROOT_PREFIX = "tvmdbg_"

class LocalCLIDebugWrapperModule(framework.BaseDebugWrapperModule):
    """Concrete subclass of BaseDebugWrapperModule implementing a local CLI.

    This class has all the methods that a `Graph Runtime` object has, in order
    to support debugging with minimal code changes. Invoking its `run()` method
    will launch the command-line interface (CLI) of tvmdbg.
    """

    def __init__(self,
                 sess,
                 graph,
                 ctx=None,
                 dump_root=None,
                 log_usage=True,
                 ui_type="curses",
                 thread_name_filter=None):
        """Constructor of LocalCLIDebugWrapperModule.

        Args:
          graph_runtime: The TVM `Graph Runtime` object being wrapped.

        Raises:
          ValueError: If dump_root is an existing and non-empty directory or if
            dump_root is a file.
        """

        self._init_command = None
        self._run_cli = None
        self._run_description = None
        self._run_info = None
        self._title = None
        self._title_color = None

        if log_usage:
            pass  # No logging for open-source.

        framework.BaseDebugWrapperModule.__init__(
            self, sess, graph, ctx, thread_name_filter=thread_name_filter)

        if not dump_root:
            self._dump_root = tempfile.mktemp(prefix=_DUMP_ROOT_PREFIX)
        else:
            dump_root = os.path.expanduser(dump_root)
            if os.path.isfile(dump_root):
                raise ValueError("dump_root path points to a file: %s" % dump_root)
            elif os.path.isdir(dump_root) and os.listdir(dump_root):
                raise ValueError("dump_root path points to a non-empty directory: %s" %
                                 dump_root)

            self._dump_root = dump_root

        self._ctx = ctx
        self._initialize_argparsers()

        # Registered tensor filters.
        self._tensor_filters = {}

        # Below are the state variables of this wrapper object.
        # _active_tensor_filter: what (if any) tensor filter is in effect. If such
        #   a filter is in effect, this object will call run() method of the
        #   underlying Tvm Session object until the filter passes. This is
        #   activated by the "-f" flag of the "run" command.
        # _run_through_times: keeps track of how many times the wrapper needs to
        #   run through without stopping at the run-end CLI. It is activated by the
        #   "-t" option of the "run" command.
        # _skip_debug: keeps track of whether the current run should be executed
        #   without debugging. It is activated by the "-n" option of the "run"
        #   command.
        #
        # _run_start_response: keeps track what OnRunStartResponse the wrapper
        #   should return at the next run-start callback. If this information is
        #   unavailable (i.e., is None), the run-start CLI will be launched to ask
        #   the user. This is the case, e.g., right before the first run starts.
        self._active_tensor_filter = None
        self._act_tensor_ftr_run_start_res = None
        self._run_through_times = 1
        self._skip_debug = False
        self._run_start_response = None
        self._is_run_start = True
        self._ui_type = ui_type
        self._graph_node_count = len(graph['nodes'])

    def _initialize_argparsers(self):
        self._argparsers = {}
        args = argparse.ArgumentParser(
            description="Run through, with or without debug tensor watching.",
            usage=argparse.SUPPRESS)
        args.add_argument(
            "-t",
            "--times",
            dest="times",
            type=int,
            default=1,
            help="How many Session.run() calls to proceed with.")
        args.add_argument(
            "-n",
            "--no_debug",
            dest="no_debug",
            action="store_true",
            help="Run through without debug tensor watching.")
        args.add_argument(
            "-f",
            "--till_filter_pass",
            dest="till_filter_pass",
            type=str,
            default="",
            help="Run until a tensor in the graph passes the specified filter.")
        args.add_argument(
            "--node_name_filter",
            dest="node_name_filter",
            type=str,
            default="",
            help="Regular-expression filter for node names to be watched in the "
                 "run, e.g., loss, reshape.*")
        args.add_argument(
            "--op_type_filter",
            dest="op_type_filter",
            type=str,
            default="",
            help="Regular-expression filter for op type to be watched in the run, "
                 "e.g., (MatMul|Add), Variable.*")
        args.add_argument(
            "--tensor_dtype_filter",
            dest="tensor_dtype_filter",
            type=str,
            default="",
            help="Regular-expression filter for tensor dtype to be watched in the "
                 "run, e.g., (float32|float64), int.*")
        self._argparsers["run"] = args
        args = argparse.ArgumentParser(
            description="Display information about this Session.run() call.",
            usage=argparse.SUPPRESS)
        self._argparsers["HOME"] = args

        self._argparsers["print_input"] = command_parser.get_view_tensor_argparser(
            "Print the value of a input in input_dict.")

    def add_tensor_filter(self, filter_name, tensor_filter):
        """Add a tensor filter.

        Args:
          filter_name: (`str`) name of the filter.
          tensor_filter: (`callable`) the filter callable. See the doc string of
            `DebugDumpDir.find()` for more details about its signature.
        """

        self._tensor_filters[filter_name] = tensor_filter

    def on_session_init(self, request):
        """Overrides on-session-init callback.

        Args:
          request: An instance of `OnSessionInitRequest`.

        Returns:
          An instance of `OnSessionInitResponse`.
        """

        return framework.OnSessionInitResponse(
            framework.OnSessionInitAction.PROCEED)

    def on_run_start(self, request):
        """Overrides on-run-start callback.

        Invoke the CLI to let user choose what action to take:
          `run` / `invoke_stepper`.

        Args:
          request: An instance of `OnRunStartRequest`.

        Returns:
          An instance of `OnRunStartResponse`.
        """
        self._is_run_start = True
        self._update_run_calls_state(
            request.run_call_count, request.outputs, request.input_dict,
            is_callable_runner=request.is_callable_runner)

        if self._active_tensor_filter:
            # If we are running until a filter passes, we just need to keep running
            # with the previous `OnRunStartResponse`.
            return self._act_tensor_ftr_run_start_res

        self._exit_if_requested_by_user()

        if self._run_call_count > 1 and not self._skip_debug:
            if self._run_through_times > 0:
                # Just run through without debugging.
                return framework.OnRunStartResponse(
                    common.CLIRunStartAction.NON_DEBUG_RUN, [])
            elif self._run_through_times == 0:
                # It is the run at which the run-end CLI will be launched: activate
                # debugging.
                return (self._run_start_response or
                        framework.OnRunStartResponse(
                            common.CLIRunStartAction.DEBUG_RUN,
                            self._get_run_debug_urls()))

        if self._run_start_response is None:
            self._prep_cli_for_run_start()

            self._run_start_response = self._launch_cli()
            if self._active_tensor_filter:
                self._act_tensor_ftr_run_start_res = self._run_start_response
            if self._run_through_times > 1:
                self._run_through_times -= 1

        self._exit_if_requested_by_user()
        return self._run_start_response

    def _exit_if_requested_by_user(self):
        if self._run_start_response == debugger_cli_common.EXPLICIT_USER_EXIT:
            # Explicit user "exit" command leads to sys.exit(1).
            print(
                "Note: user exited from debugger CLI: Calling sys.exit(1).",
                file=sys.stderr)
            sys.exit(1)

    def _prep_cli_for_run_start(self):
        """Prepare (but not launch) the CLI for run-start."""

        self._run_cli = ui_factory.get_ui(self._ui_type)

        help_intro = debugger_cli_common.RichTextLines([])
        if self._run_call_count == 1:
            # Show logo at the onset of the first run.
            help_intro.extend(cli_shared.get_tvmdbg_logo())
        help_intro.extend(debugger_cli_common.RichTextLines("Upcoming run:"))
        help_intro.extend(self._run_info)

        self._run_cli.set_help_intro(help_intro)

        # Create initial screen output detailing the run.
        self._title = "run-start: " + self._run_description
        self._init_command = "HOME"
        self._title_color = "blue_on_white"

    def on_run_end(self, request):
        """Overrides on-run-end callback.

        Actions taken:
          1) Load the debug dump.
          2) Bring up the Analyzer CLI.

        Args:
          request: An instance of OnSessionInitRequest.

        Returns:
          An instance of OnSessionInitResponse.
        """

        self._is_run_start = False
        if request.performed_action == common.CLIRunStartAction.DEBUG_RUN:
            partition_graphs = None
            if request.tvm_error and not os.path.isdir(self._dump_root):
                # It is possible that the dump root may not exist due to errors that
                # have occurred prior to graph execution (e.g., invalid device
                # assignments), in which case we will just raise the exception as the
                # unwrapped Session does.
                raise request.tvm_error

            debug_dump = debug_data.DebugDumpDir(self._ctx,
                                                 self._dump_root, partition_graphs=partition_graphs)

            passed_filter = None
            if self._active_tensor_filter:
                if not debug_dump.find(
                        self._tensor_filters[self._active_tensor_filter], first_n=1):
                    # No dumped tensor passes the filter in this run. Clean up the dump
                    # directory and move on.
                    self._remove_dump_root()
                    return framework.OnRunEndResponse()
                else:
                    # Some dumped tensor(s) from this run passed the filter.
                    passed_filter = self._active_tensor_filter
                    self._active_tensor_filter = None

            self._prep_debug_cli_for_run_end(
                debug_dump, request.tvm_error, passed_filter)

            self._run_start_response = self._launch_cli()

            # Clean up the dump generated by this run.
            self._remove_dump_root()
        else:
            # No debug information to show following a non-debug run() call.
            self._run_start_response = None

        # Return placeholder response that currently holds no additional
        # information.
        return framework.OnRunEndResponse()

    def _remove_dump_root(self):
        if os.path.isdir(self._dump_root):
            shutil.rmtree(self._dump_root)

    def _prep_debug_cli_for_run_end(self, debug_dump, tvm_error, passed_filter):
        """Prepare (but not launch) CLI for run-end, with debug dump from the run.

        Args:
          debug_dump: (debug_data.DebugDumpDir) The debug dump directory from this
            run.
          tvm_error: (None or OpError) OpError that happened during the run() call
            (if any).
          passed_filter: (None or str) Name of the tensor filter that just passed
            and caused the preparation of this run-end CLI (if any).
        """

        if tvm_error:
            help_intro = cli_shared.get_error_intro(tvm_error)

            self._init_command = "help"
            self._title_color = "red_on_white"
        else:
            help_intro = None
            self._init_command = "lt"

            self._title_color = "black_on_white"
            if passed_filter is not None:
                # Some dumped tensor(s) from this run passed the filter.
                self._init_command = "lt -f %s" % passed_filter
                self._title_color = "red_on_white"

        self._run_cli = analyzer_cli.create_analyzer_ui(
            debug_dump, self._tensor_filters, ui_type=self._ui_type,
            on_ui_exit=self._remove_dump_root)

        # Get names of all dumped tensors.
        dumped_tensor_names = []
        for datum in debug_dump.dumped_tensor_data:
            dumped_tensor_names.append("%s:%d" %
                                       (datum.node_name, datum.output_slot))

        # Tab completions for command "view_tensors".
        self._run_cli.register_tab_comp_context(["view_tensor", "pt"],
                                                dumped_tensor_names)

        # Tab completion for commands "node_details", "graphnode_inputs" and
        # "graphnode_outputs". The list comprehension is used below because nodes()
        # output can be unicodes and they need to be converted to strs.
        self._run_cli.register_tab_comp_context(
            ["node_details", "ni", "graphnode_inputs", "li", "graphnode_outputs", "lo"],
            [str(node_name) for node_name in debug_dump.nodes()])
        # TODO(cais): Reduce API surface area for aliases vis-a-vis tab
        #    completion contexts and registered command handlers.

        self._title = "run-end: " + self._run_description

        if help_intro:
            self._run_cli.set_help_intro(help_intro)

    def _launch_cli(self):
        """Launch the interactive command-line interface.

        Returns:
          The OnRunStartResponse specified by the user using the "run" command.
        """

        self._register_this_run_info(self._run_cli)
        response = self._run_cli.run_ui(
            init_command=self._init_command,
            title=self._title,
            title_color=self._title_color)

        return response

    def _run_info_handler(self, args, screen_info=None):
        _ = args  # Currently unused.
        _ = screen_info  # Currently unused.
        output = debugger_cli_common.RichTextLines([])

        if self._run_call_count == 1:
            output.extend(cli_shared.get_tvmdbg_logo())
        output.extend(self._run_info)

        if (not self._is_run_start and
                debugger_cli_common.MAIN_MENU_KEY in output.annotations):
            menu = output.annotations[debugger_cli_common.MAIN_MENU_KEY]
            if "list_graphnodes" not in menu.captions():
                menu.insert(
                    0, debugger_cli_common.MenuItem("list_graphnodes", "list_graphnodes"))

        return output

    def _print_input_handler(self, args, screen_info=None):
        np_printoptions = cli_shared.get_np_printoptions_frm_scr(
            screen_info)

        if not self._input_dict:
            return cli_shared.error(
                "The input_dict of the current run is None or empty.")

        parsed = self._argparsers["print_input"].parse_args(args)
        tensor_name, tensor_slicing = (
            command_parser.parse_tensor_name_with_slicing(parsed.tensor_name))

        input_key = None
        input_value = None
        for key in self._input_dict:
            key_name = common.get_graph_element_name(key)
            if key_name == tensor_name:
                input_key = key_name
                input_value = self._input_dict[key]
                break

        if input_key is None:
            return cli_shared.error(
                "The input_dict of the current run does not contain the key %s" %
                tensor_name)
        return cli_shared.format_tensor(
            input_value,
            input_key + " (input)",
            np_printoptions,
            print_all=parsed.print_all,
            tensor_slicing=tensor_slicing,
            highlight_options=cli_shared.parse_ranges_highlight(parsed.ranges),
            include_numeric_summary=parsed.numeric_summary)

    def _run_handler(self, args, screen_info=None):
        """Command handler for "run" command during on-run-start."""

        del screen_info  # Currently unused.

        parsed = self._argparsers["run"].parse_args(args)
        parsed.node_name_filter = parsed.node_name_filter or None
        parsed.op_type_filter = parsed.op_type_filter or None
        parsed.tensor_dtype_filter = parsed.tensor_dtype_filter or None

        self._skip_debug = parsed.no_debug
        self._run_through_times = parsed.times

        if parsed.times > 1 or parsed.no_debug:
            # If requested -t times > 1, the very next run will be a non-debug run.
            action = common.CLIRunStartAction.NON_DEBUG_RUN
            debug_urls = []
        else:
            action = common.CLIRunStartAction.DEBUG_RUN
            debug_urls = self._get_run_debug_urls()
        run_start_response = framework.OnRunStartResponse(
            action,
            debug_urls,
            node_name_regex_whitelist=parsed.node_name_filter,
            op_type_regex_whitelist=parsed.op_type_filter,
            tensor_dtype_regex_whitelist=parsed.tensor_dtype_filter)

        if parsed.till_filter_pass:
            # For the run-till-filter-pass (run -f) mode, use the DEBUG_RUN
            # option to access the intermediate tensors, and set the corresponding
            # state flag of the class itself to True.
            if parsed.till_filter_pass in self._tensor_filters:
                action = common.CLIRunStartAction.DEBUG_RUN
                self._active_tensor_filter = parsed.till_filter_pass
                self._act_tensor_ftr_run_start_res = run_start_response
            else:
                # Handle invalid filter name.
                return debugger_cli_common.RichTextLines(
                    ["ERROR: tensor filter \"%s\" does not exist." %
                     parsed.till_filter_pass])

        # Raise CommandLineExit exception to cause the CLI to exit.
        raise debugger_cli_common.CommandLineExit(exit_token=run_start_response)

    def _register_this_run_info(self, curses_cli):
        curses_cli.register_command_handler(
            "run",
            self._run_handler,
            self._argparsers["run"].format_help(),
            prefix_aliases=["r"])
        curses_cli.register_command_handler(
            "HOME",
            self._run_info_handler,
            self._argparsers["HOME"].format_help(),
            prefix_aliases=["H"])
        curses_cli.register_command_handler(
            "home",
            self._run_info_handler,
            self._argparsers["HOME"].format_help(),
            prefix_aliases=["ho"])
        curses_cli.register_command_handler(
            "print_input",
            self._print_input_handler,
            self._argparsers["print_input"].format_help(),
            prefix_aliases=["pf"])

        if self._tensor_filters:
            # Register tab completion for the filter names.
            curses_cli.register_tab_comp_context(["run", "r"],
                                                 list(self._tensor_filters.keys()))
        if self._input_dict:
            # Register tab completion for input_dict keys.
            input_keys = [common.get_graph_element_name(key)
                          for key in self._input_dict.keys()]
            curses_cli.register_tab_comp_context(["print_input", "pf"], input_keys)

    def _get_run_debug_urls(self):
        """Get the debug_urls value for the current run() call.

        Returns:
          debug_urls: (list of str) Debug URLs for the current run() call.
            Currently, the list consists of only one URL that is a file:// URL.
        """

        return ["file://" + self._dump_root]

    def _update_run_calls_state(self,
                                run_call_count,
                                outputs,
                                input_dict,
                                is_callable_runner=False):
        """Update the internal state with regard to run() call history.

        Args:
          run_call_count: (int) Number of run() calls that have occurred.
          outputs: a node/tensor or a list of node/tensor that are the outputs of
            the run() call. This is the same as the outputs argument to the run()
            call.
          input_dict: None of a dict. This is the input_dict argument to the run()
            call.
          is_callable_runner: (bool) whether a runner returned by
            Session.make_callable is being run.
        """

        self._run_call_count = run_call_count
        self._input_dict = input_dict
        self._run_description = cli_shared.get_run_short_description(
            run_call_count,
            outputs,
            input_dict,
            is_callable_runner=is_callable_runner)
        self._run_through_times -= 1

        self._run_info = cli_shared.get_run_start_intro(
            run_call_count,
            self._graph_node_count,
            outputs,
            input_dict,
            self._tensor_filters,
            is_callable_runner=is_callable_runner)
