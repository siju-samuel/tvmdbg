from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import re
import threading
import logging

#from core.protobuf import config_pb2
#from python.client import session
#from python.framework import errors
#from python.framework import ops
#from python.training import monitored_session
#from python.util import nest

from ..util import debug_utils
#from python.debug.lib import stepper
from ..util import common
from ..examples import debug_v
def PRINT(txt=""):
    debug_v.PRINT(txt)

# Helper function.
def _check_type(obj, expected_types):
  """Check if an object is of the expected type.

  Args:
    obj: The object being checked.
    expected_types: (`type` or an iterable of `type`s) The expected `type`(s)
      of obj.

  Raises:
      TypeError: If obj is not an instance of expected_type.
  """
  PRINT()
  if not isinstance(obj, expected_types):
    raise TypeError("Expected type %s; got type %s" %
                    (expected_types, type(obj)))


class OnSessionInitRequest(object):
  """Request to an on-session-init callback.

  This callback is invoked during the __init__ call to a debug-wrapper session.
  """

  def __init__(self, sess):
    """Constructor.

    Args:
      sess: A TVM Session object.
    """
    PRINT()
    #_check_type(sess, (session.BaseSession, monitored_session.MonitoredSession))
    PRINT("ERROR :: _check_type(sess, (session.BaseSession, monitored_session.MonitoredSession)) need to implement")
    self.session = sess


class OnSessionInitAction(object):
  """Enum-like values for possible action to take on session init."""

  # Proceed, without special actions, in the wrapper session initialization.
  # What action the wrapper session performs next is determined by the caller
  # of the wrapper session. E.g., it can call run().
  PROCEED = "proceed"

  # Instead of letting the caller of the wrapper session determine what actions
  # the wrapper session will perform next, enter a loop to receive instructions
  # from a remote client.
  # For example, TensorBoard visual debugger can use this action so that it can
  # launch session.run() calls remotely.
  REMOTE_INSTR_LOOP = "remote_instr_loop"


class OnSessionInitResponse(object):
  """Response from an on-session-init callback."""

  def __init__(self, action):
    """Constructor.

    Args:
      action: (`OnSessionInitAction`) Debugger action to take on session init.
    """
    PRINT()
    _check_type(action, str)
    self.action = action


class OnRunStartRequest(object):
  """Request to an on-run-start callback.

  This callback is invoked during a run() call of the debug-wrapper
  session, immediately after the run() call counter is incremented.
  """

  def __init__(self, fetches, feed_dict, run_options, run_metadata,
               run_call_count, is_callable_runner=False):
    """Constructor of `OnRunStartRequest`.

    Args:
      fetches: Fetch targets of the run() call.
      feed_dict: The feed dictionary to the run() call.
      run_options: RunOptions input to the run() call.
      run_metadata: RunMetadata input to the run() call.
        The above four arguments are identical to the input arguments to the
        run() method of a non-wrapped TVM session.
      run_call_count: 1-based count of how many run calls (including this one)
        has been invoked.
      is_callable_runner: (bool) whether a runner returned by
        Session.make_callable is being run.
    """
    PRINT()
    self.fetches = fetches
    self.feed_dict = feed_dict
    self.run_options = run_options
    self.run_metadata = run_metadata
    self.run_call_count = run_call_count
    self.is_callable_runner = is_callable_runner


class OnRunStartAction(object):
  """Enum-like values for possible action to take on start of a run() call."""

  # Run once with debug tensor-watching.
  DEBUG_RUN = "debug_run"

  # Run once with profiler.
  PROFILE_RUN = "profile_run"

  # Run without debug tensor-watching.
  NON_DEBUG_RUN = "non_debug_run"

  # Instead of running the fetches as a whole, as would normally happen, invoke
  # the (to-be-implemented) debug stepper.
  # TODO(cais): Remove "to-be-implemented".
  INVOKE_STEPPER = "invoke_stepper"


class OnRunStartResponse(object):
  """Request from an on-run-start callback.

  The caller of the callback can use this response object to specify what
  action the debug-wrapper session actually takes on the run() call.
  """

  def __init__(self,
               action,
               debug_urls,
               debug_ops="DebugIdentity",
               node_name_regex_whitelist=None,
               op_type_regex_whitelist=None,
               tensor_dtype_regex_whitelist=None,
               tolerate_debug_op_creation_failures=False):
    """Constructor of `OnRunStartResponse`.

    Args:
      action: (`OnRunStartAction`) the action actually taken by the wrapped
        session for the run() call.
      debug_urls: (`list` of `str`) debug_urls used in watching the tensors
        during the run() call.
      debug_ops: (`str` or `list` of `str`) Debug op(s) to be used by the
        debugger.
      node_name_regex_whitelist: Regular-expression whitelist for node
        name.
      op_type_regex_whitelist: Regular-expression whitelist for op type.
      tensor_dtype_regex_whitelist: Regular-expression whitelist for tensor
        dtype.
      tolerate_debug_op_creation_failures: Whether debug op creation failures
        are to be tolerated.
    """
    PRINT()

    _check_type(action, str)
    self.action = action

    _check_type(debug_urls, list)
    self.debug_urls = debug_urls

    self.debug_ops = debug_ops

    self.node_name_regex_whitelist = node_name_regex_whitelist
    self.op_type_regex_whitelist = op_type_regex_whitelist
    self.tensor_dtype_regex_whitelist = tensor_dtype_regex_whitelist
    self.tolerate_debug_op_creation_failures = (
        tolerate_debug_op_creation_failures)


class OnRunEndRequest(object):
  """Request to an on-run-end callback.

  The callback is invoked immediately before the wrapped run() call ends.
  """

  def __init__(self,
               performed_action,
               run_metadata=None,
               client_graph_def=None,
               tvm_error=None):
    """Constructor for `OnRunEndRequest`.

    Args:
      performed_action: (`OnRunStartAction`) Actually-performed action by the
        debug-wrapper session.
      run_metadata: run_metadata output from the run() call (if any).
      client_graph_def: (GraphDef) GraphDef from the client side, i.e., from
        the python front end of TVM. Can be obtained with
        session.graph.as_graph_def().
      tf_error: (errors.OpError subtypes) TVM OpError that occurred
        during the run (if any).
    """
    PRINT()

    _check_type(performed_action, str)
    self.performed_action = performed_action

    if run_metadata is not None:
      #_check_type(run_metadata, config_pb2.RunMetadata)
      PRINT("ERROR :: _check_type(run_metadata, config_pb2.RunMetadata) need to implement")
    self.run_metadata = run_metadata
    self.client_graph_def = client_graph_def
    self.tvm_error = tvm_error


class OnRunEndResponse(object):
  """Response from an on-run-end callback."""

  def __init__(self):
    PRINT()

    # Currently only a placeholder.
    pass


#class BaseDebugWrapperSession(session.SessionInterface):
class BaseDebugWrapperSession():
  """Base class of debug-wrapper session classes.

  Concrete classes that inherit from this class need to implement the abstract
  methods such as on_session_init, on_run_start and on_run_end.
  """

  # TODO(cais): Add on_cont_start and on_cont_end callbacks once the stepper is
  # is available.

  def __init__(self, sess, graph, thread_name_filter=None,
               pass_through_operrors=False):
    """Constructor of `BaseDebugWrapperSession`.

    Args:
      sess: An (unwrapped) TVM session instance. It should be a subtype
        of `BaseSession` or `tf.MonitoredSession`.
      thread_name_filter: Regular-expression filter (whitelist) for name(s) of
        thread(s) on which the wrapper session will be active. This regular
        expression is used in a start-anchored fashion on the thread name, i.e.,
        by applying the `match` method of the compiled pattern. The default
        `None` means that the wrapper session will be active on all threads.
        E.g., r"MainThread$", r"QueueRunnerThread.*".
      pass_through_operrors: If True, all captured OpErrors will be
        propagated.  By default this captures all OpErrors.

    Raises:
      ValueError: On invalid `OnSessionInitAction` value.
      NotImplementedError: If a non-DirectSession sess object is received.
    """
    PRINT()
    PRINT("ERROR :: session.SessionInterface need to implement")
    #PRINT(graph.json())
    self._feed_dict = {}
    self._graph = graph

    #_check_type(sess, (session.BaseSession, monitored_session.MonitoredSession))
    PRINT("ERROR :: Session check need to implement")

    # The session being wrapped.
    self._sess = sess
    self._thread_name_filter_pattern = (re.compile(thread_name_filter)
                                        if thread_name_filter else None)
    # TODO(cais/kstevens): Unittest this pass through feature.
    self._pass_through_operrors = pass_through_operrors

    # Keeps track of number of run calls that have been performed on this
    # debug-wrapper session. The count can be used for purposes such as
    # displaying the state of the Session in a UI and determining a run
    # number-dependent debug URL.
    self._run_call_count = 0

    # Invoke on-session-init callback.
    response = self.on_session_init(OnSessionInitRequest(self._sess))
    _check_type(response, OnSessionInitResponse)

    if response.action == OnSessionInitAction.PROCEED:
      pass
    elif response.action == OnSessionInitAction.REMOTE_INSTR_LOOP:
      # TODO(cais): Implement REMOTE_INSTR_LOOP
      raise NotImplementedError(
          "OnSessionInitAction REMOTE_INSTR_LOOP has not been "
          "implemented.")
    else:
      raise ValueError(
          "Invalid OnSessionInitAction value: %s" % response.action)

    self._default_session_context_manager = None

  @property
  def graph(self):
    #return self._sess.graph
    PRINT("ERROR :: self._sess.graph need to implement")
    pass

  @property
  def graph_def(self):
    #return self._sess.graph_def
    PRINT("ERROR :: self._sess.graph_def need to implement")
    pass

  @property
  def sess_str(self):
    #return self._sess.sess_str
    PRINT("ERROR :: self._sess.sess_str need to implement")
    pass

  @property
  def session(self):
    return self._sess


  def set_input(self, name, value):
    self._feed_dict.update({name:value})

  def run(self,
          fetches,
          options=None,
          run_metadata=None,
          callable_runner=None,
          callable_runner_args=None):
    """Wrapper around Session.run() that inserts tensor watch options.

    Args:
      fetches: Same as the `fetches` arg to regular `Session.run()`.
      options: Same as the `options` arg to regular `Session.run()`.
      run_metadata: Same as the `run_metadata` arg to regular `Session.run()`.
      callable_runner: A `callable` returned by `Session.make_callable()`.
        If not `None`, `fetches` and `feed_dict` must both be `None`.
      callable_runner_args: An optional list of arguments to `callable_runner`.

    Returns:
      Simply forwards the output of the wrapped `Session.run()` call.

    Raises:
      ValueError: On invalid `OnRunStartAction` value. Or if `callable_runner`
        is not `None` and either or both of `fetches` and `feed_dict` is `None`.
    """
    PRINT()
    fetches = self._sess.get_output_names()

    if not callable_runner:
      self.increment_run_call_count()
    else:
      if fetches or self._feed_dict:
        raise ValueError(
            "callable_runner and fetches/feed_dict are mutually exclusive, but "
            "are used simultaneously.")

    if self._is_disabled_thread():
      if callable_runner:
        return callable_runner(*callable_runner_args)
      else:
        self._sess.run(self._feed_dict)
        return True

    # Invoke on-run-start callback and obtain response.
    run_start_resp = self.on_run_start(
        OnRunStartRequest(self._sess.get_output_names(), self._feed_dict, options, run_metadata,
                          self._run_call_count,
                          is_callable_runner=bool(callable_runner)))
    _check_type(run_start_resp, OnRunStartResponse)

    if run_start_resp.action == OnRunStartAction.DEBUG_RUN:
      retvals = True
      # Decorate RunOption to fill in debugger tensor watch specifications.
      PRINT("ERROR :: config_pb2.RunOptions() need to implement")
      decorated_run_options = options # or config_pb2.RunOptions()
      PRINT("ERROR :: config_pb2.RunMetadata() need to implement")
      run_metadata = run_metadata# or config_pb2.RunMetadata()

      """self._decorate_run_options_for_debug(
          decorated_run_options,
          run_start_resp.debug_urls,
          debug_ops=run_start_resp.debug_ops,
          node_name_regex_whitelist=run_start_resp.node_name_regex_whitelist,
          op_type_regex_whitelist=run_start_resp.op_type_regex_whitelist,
          tensor_dtype_regex_whitelist=(
              run_start_resp.tensor_dtype_regex_whitelist),
          tolerate_debug_op_creation_failures=(
              run_start_resp.tolerate_debug_op_creation_failures))"""

      # Invoke the run() method of the wrapped Session. Catch any TVM
      # runtime errors.
      tvm_error = None
      try:
        if callable_runner:
          retvals = callable_runner(*callable_runner_args,
                                    options=decorated_run_options,
                                    run_metadata=run_metadata)
        else:
          retvals = self._sess.debugRun()
      except errors.OpError as op_error:
        if self._pass_through_operrors:
          raise op_error
        tvm_error = op_error
        retvals = op_error
      PRINT("ERROR :: self._sess.run need to implement")

      PRINT("ERROR :: self._sess.graph.as_graph_def() need to implement")
      run_end_req = OnRunEndRequest(
          run_start_resp.action,
          run_metadata=run_metadata,
          #client_graph_def=self._sess.graph.as_graph_def(),
          client_graph_def="",
          tvm_error=tvm_error)

    elif run_start_resp.action == OnRunStartAction.PROFILE_RUN:
      PRINT("ERROR :: config_pb2.RunOptions() need to implement")
      decorated_run_options = options# or config_pb2.RunOptions()
      PRINT("ERROR :: config_pb2.RunMetadata() need to implement")
      run_metadata = run_metadata# or config_pb2.RunMetadata()
      self._decorate_run_options_for_profile(decorated_run_options)
      if callable_runner:
        retvals = callable_runner(*callable_runner_args,
                                  options=decorated_run_options,
                                  run_metadata=run_metadata)
      else:
        PRINT("ERROR :: self._sess.run() need to implement")
        retvals = False
        #retvals = self._sess.run(fetches,
        #                         feed_dict=feed_dict,
        #                         options=decorated_run_options,
        #                         run_metadata=run_metadata)
      PRINT("ERROR :: self._sess.graph.as_graph_def() need to implement")
      run_end_req = OnRunEndRequest(
          run_start_resp.action,
          run_metadata=run_metadata,
          #client_graph_def=self._sess.graph.as_graph_def())
          client_graph_def="")
    elif (run_start_resp.action == OnRunStartAction.NON_DEBUG_RUN or
          run_start_resp.action == OnRunStartAction.INVOKE_STEPPER):
      if callable_runner:
        raise NotImplementedError(
            "Stepper mode is not implemented for callables created by "
            "Session.make_callable().")

      if run_start_resp.action == OnRunStartAction.INVOKE_STEPPER:
        PRINT("ERROR :: stepper.NodeStepper is not implemented")
        raise NotImplementedError(
            "stepper.NodeStepper is not implemented")
        #with stepper.NodeStepper(
        #    self._sess, fetches, feed_dict) as node_stepper:
        #  retvals = self.invoke_node_stepper(
        #      node_stepper, restore_variable_values_on_exit=True)

      # Invoke run() method of the wrapped session.
      PRINT("ERROR :: self._sess.reset(*args, **kwargs) need to implement")
      retvals = False
      #retvals = self._sess.run(
      #    fetches,
      #    feed_dict=feed_dict,
      #    options=options,
      #    run_metadata=run_metadata)

      # Prepare arg for the on-run-end callback.
      run_end_req = OnRunEndRequest(run_start_resp.action)
    else:
      raise ValueError(
          "Invalid OnRunStartAction value: %s" % run_start_resp.action)

    # Invoke on-run-end callback and obtain response.
    run_end_resp = self.on_run_end(run_end_req)
    _check_type(run_end_resp, OnRunEndResponse)
    # Currently run_end_resp is only a placeholder. No action is taken on it.

    return retvals

  def _is_disabled_thread(self):
    PRINT()
    thread_name = threading.current_thread().name or ""
    return (self._thread_name_filter_pattern and
            not self._thread_name_filter_pattern.match(thread_name))

  def run_step_fn(self, step_fn):
    PRINT()
    PRINT("ERROR :: step_fn need to implement")
    #return step_fn(
    #    monitored_session.MonitoredSession.StepContext(self._sess, self.run))
    raise NotImplementedError(
        "step_fn is not implemented for debug-wrapper sessions.")

  def partial_run_setup(self, fetches, feeds=None):
    """Sets up the feeds and fetches for partial runs in the session."""
    raise NotImplementedError(
        "partial_run_setup is not implemented for debug-wrapper sessions.")

  def partial_run(self, handle, fetches, feed_dict=None):
    raise NotImplementedError(
        "partial_run is not implemented for debug-wrapper sessions.")

  def list_devices(self, *args, **kwargs):
    #return self._sess.list_devices(*args, **kwargs)
    PRINT("ERROR :: self._sess.list_devices(*args, **kwargs) need to implement")
    pass

  def reset(self, *args, **kwargs):
    #return self._sess.reset(*args, **kwargs)
    PRINT("ERROR :: self._sess.reset(*args, **kwargs) need to implement")
    pass

  def make_callable(self,
                    fetches,
                    feed_list=None,
                    accept_options=False):
    PRINT()
    PRINT("ERROR :: self._sess.make_callable need to implement")
    #runner = self._sess.make_callable(
    #    fetches, feed_list=feed_list, accept_options=True)
    #def wrapped_runner(*runner_args, **kwargs):
    #  return self.run(None,
    #                  feed_dict=None,
    #                  options=kwargs.get("options", None),
    #                  run_metadata=kwargs.get("run_metadata", None),
    #                  callable_runner=runner,
    #                  callable_runner_args=runner_args)

    #return wrapped_runner
    pass

  @property
  def run_call_count(self):
    return self._run_call_count

  def increment_run_call_count(self):
    self._run_call_count += 1

  def _decorate_run_options_for_debug(
      self,
      run_options,
      debug_urls,
      debug_ops="DebugIdentity",
      node_name_regex_whitelist=None,
      op_type_regex_whitelist=None,
      tensor_dtype_regex_whitelist=None,
      tolerate_debug_op_creation_failures=False):
    """Modify a RunOptions object for debug tensor watching.

    Specifies request for outputting partition graphs. Adds
    debug_tensor_watch_opts with proper debug URLs.

    Args:
      run_options: (RunOptions) the modified RunOptions object.
      debug_urls: (list of str) debug URLs to be entered in run_options.
        debug_tensor_watch_opts.
      debug_ops: (str or list of str) debug op(s) to be used by the debugger.
      node_name_regex_whitelist: Regular-expression whitelist for node
        name.
      op_type_regex_whitelist: Regular-expression whitelist for op type.
      tensor_dtype_regex_whitelist: Regular-expression whitelist for tensor
        dtype.
      tolerate_debug_op_creation_failures: Whether debug op creation failures
        are to be tolerated.
    """
    PRINT()

    PRINT("ERROR :: self._sess.graph() need to implement")
    print("debug_urls:", debug_urls)
    run_options.output_partition_graphs = True
    #debug_utils.watch_graph(
    #    run_options,
    #    self._sess.graph,
    #    debug_urls=debug_urls,
    #    debug_ops=debug_ops,
    #    node_name_regex_whitelist=node_name_regex_whitelist,
    #    op_type_regex_whitelist=op_type_regex_whitelist,
    #    tensor_dtype_regex_whitelist=tensor_dtype_regex_whitelist,
    #    tolerate_debug_op_creation_failures=tolerate_debug_op_creation_failures)

  def _decorate_run_options_for_profile(self, run_options):
    """Modify a RunOptions object for profiling TVM graph execution.

    Args:
      run_options: (RunOptions) the modified RunOptions object.
    """

    PRINT("ERROR :: run_options.trace_level = config_pb2.RunOptions.FULL_TRACE need to implement")
    #run_options.trace_level = config_pb2.RunOptions.FULL_TRACE
    run_options.trace_level = 0

  @abc.abstractmethod
  def on_session_init(self, request):
    """Callback invoked during construction of the debug-wrapper session.

    This is a blocking callback.
    The invocation happens right before the constructor ends.

    Args:
      request: (`OnSessionInitRequest`) callback request carrying information
        such as the session being wrapped.

    Returns:
      An instance of `OnSessionInitResponse`.
    """

  @abc.abstractmethod
  def on_run_start(self, request):
    """Callback invoked on run() calls to the debug-wrapper session.

    This is a blocking callback.
    The invocation happens after the wrapper's run() call is entered,
    after an increment of run call counter.

    Args:
      request: (`OnRunStartRequest`) callback request object carrying
        information about the run call such as the fetches, feed dict, run
        options, run metadata, and how many `run()` calls to this wrapper
        session have occurred.

    Returns:
      An instance of `OnRunStartResponse`, carrying information to
        1) direct the wrapper session to perform a specified action (e.g., run
          with or without debug tensor watching, invoking the stepper.)
        2) debug URLs used to watch the tensors.
    """

  @abc.abstractmethod
  def on_run_end(self, request):
    """Callback invoked on run() calls to the debug-wrapper session.

    This is a blocking callback.
    The invocation happens right before the wrapper exits its run() call.

    Args:
      request: (`OnRunEndRequest`) callback request object carrying information
        such as the actual action performed by the session wrapper for the
        run() call.

    Returns:
      An instance of `OnRunStartResponse`.
    """

  def as_default(self):
    #return ops.default_session(self)
    PRINT("ERROR :: ops.default_session/python tvm session need to implement")
    raise NotImplementedError(
        "ops.default_session(self) is not implemented for debug-wrapper sessions.")

  def __enter__(self):
    if self._default_session_context_manager is None:
      self._default_session_context_manager = self.as_default()
    return self._default_session_context_manager.__enter__()

  def __exit__(self, exec_type, exec_value, exec_tb):
    self._default_session_context_manager.__exit__(
        exec_type, exec_value, exec_tb)

  def __del__(self):
    if hasattr(self._sess, "__del__"):
      #self._sess.__del__()
      PRINT("ERROR :: self._sess.__del__() need to implement")

  def close(self):
    #self._sess.close()
    PRINT("ERROR :: self._sess.close() need to implement")

  # TODO(cais): Add _node_name_regex_whitelist and
  #   _node_op_type_regex_whitelist.

  @abc.abstractmethod
  def invoke_node_stepper(self,
                          node_stepper,
                          restore_variable_values_on_exit=True):
    """Callback invoked when the client intends to step through graph nodes.

    Args:
      node_stepper: (stepper.NodeStepper) An instance of NodeStepper to be used
        in this stepping session.
      restore_variable_values_on_exit: (bool) Whether any variables whose values
        have been altered during this node-stepper invocation should be restored
        to their old values when this invocation ends.

    Returns:
      The same return values as the `Session.run()` call on the same fetches as
        the NodeStepper.
    """

  def should_stop(self):
    PRINT()
    if hasattr(self._sess, "should_stop"):
      #return self._sess.should_stop()
      PRINT("ERROR :: self._sess.should_stop() need to implement")
      pass
    else:
      raise ValueError(
          "The wrapped session %r does not have a method called 'should_stop'. "
          "Do you intend to wrap a tf.MonitoredSession instead?" % self._sess)


class WatchOptions(object):
  """Type for return values of watch_fn."""

  def __init__(self,
               debug_ops=None,
               node_name_regex_whitelist=None,
               op_type_regex_whitelist=None,
               tensor_dtype_regex_whitelist=None,
               tolerate_debug_op_creation_failures=False):
    """Constructor of WatchOptions: Debug watch options.

    Used as return values of `watch_fn`s.

    Args:
      debug_ops: (`str` or `list of str`) Debug ops to be used.
      node_name_regex_whitelist: Regular-expression whitelist for node_name,
        e.g., `"(weight_[0-9]+|bias_.*)"`
      op_type_regex_whitelist: Regular-expression whitelist for the op type of
        nodes, e.g., `"(Variable|Add)"`.
        If both `node_name_regex_whitelist` and `op_type_regex_whitelist`
        are set, the two filtering operations will occur in a logical `AND`
        relation. In other words, a node will be included if and only if it
        hits both whitelists.
      tensor_dtype_regex_whitelist: Regular-expression whitelist for Tensor
        data type, e.g., `"^int.*"`.
        This whitelist operates in logical `AND` relations to the two whitelists
        above.
      tolerate_debug_op_creation_failures: (`bool`) whether debug op creation
        failures (e.g., due to dtype incompatibility) are to be tolerated by not
        throwing exceptions.
    """
    PRINT()
    if debug_ops:
      self.debug_ops = debug_ops
    else:
      self.debug_ops = ["DebugIdentity"]
    self.node_name_regex_whitelist = node_name_regex_whitelist
    self.op_type_regex_whitelist = op_type_regex_whitelist
    self.tensor_dtype_regex_whitelist = tensor_dtype_regex_whitelist
    self.tolerate_debug_op_creation_failures = (
        tolerate_debug_op_creation_failures)

  def __repr__(self):
    return ("WatchOptions(debug_ops=%r, node_name_regex_whitelist=%r, "
            "op_type_regex_whitelist=%r, tensor_dtype_regex_whitelist=%r, "
            "tolerate_debug_op_creation_failures=%r)" % (
                self.debug_ops, self.node_name_regex_whitelist,
                self.op_type_regex_whitelist, self.tensor_dtype_regex_whitelist,
                self.tolerate_debug_op_creation_failures))


class NonInteractiveDebugWrapperSession(BaseDebugWrapperSession):
  """Base class for non-interactive (i.e., non-CLI) debug wrapper sessions."""

  def __init__(self, sess, watch_fn=None, thread_name_filter=None,
               pass_through_operrors=False):
    """Constructor of NonInteractiveDebugWrapperSession.

    Args:
      sess: The TVM `Session` object being wrapped.
      watch_fn: (`Callable`) A Callable that maps the fetches and feeds of a
        debugged `Session.run()` call to `WatchOptions.`
        * Args:
          * `fetches`: the fetches to the `Session.run()` call.
          * `feeds`: the feeds to the `Session.run()` call.

        * Returns:
         (`tf_debug.WatchOptions`) An object containing debug options including
           the debug ops to use, the node names, op types and/or tensor data
           types to watch, etc. See the documentation of `tf_debug.WatchOptions`
           for more details.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      pass_through_operrors: If true, all captured OpErrors will be
        propagated.  By default this captures all OpErrors.
    Raises:
       TypeError: If a non-None `watch_fn` is specified and it is not callable.
    """
    PRINT()

    BaseDebugWrapperSession.__init__(
        self, sess, thread_name_filter=thread_name_filter,
        pass_through_operrors=pass_through_operrors)

    self._watch_fn = None
    if watch_fn is not None:
      if not callable(watch_fn):
        raise TypeError("watch_fn is not callable")
      self._watch_fn = watch_fn

  def on_session_init(self, request):
    """See doc of BaseDebugWrapperSession.on_run_start."""

    return OnSessionInitResponse(OnSessionInitAction.PROCEED)

  @abc.abstractmethod
  def prepare_run_debug_urls(self, fetches, feed_dict):
    """Abstract method to be implemented by concrete subclasses.

    This method prepares the run-specific debug URL(s).

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`
      feed_dict: Same as the `feed_dict` argument to `Session.run()`

    Returns:
      debug_urls: (`str` or `list` of `str`) Debug URLs to be used in
        this `Session.run()` call.
    """

  def on_run_start(self, request):
    """See doc of BaseDebugWrapperSession.on_run_start."""
    PRINT()

    debug_urls, watch_opts = self._prepare_run_watch_config(
        request.fetches, request.feed_dict)

    return OnRunStartResponse(
        OnRunStartAction.DEBUG_RUN,
        debug_urls,
        debug_ops=watch_opts.debug_ops,
        node_name_regex_whitelist=watch_opts.node_name_regex_whitelist,
        op_type_regex_whitelist=watch_opts.op_type_regex_whitelist,
        tensor_dtype_regex_whitelist=watch_opts.tensor_dtype_regex_whitelist,
        tolerate_debug_op_creation_failures=(
            watch_opts.tolerate_debug_op_creation_failures))

  def _prepare_run_watch_config(self, fetches, feed_dict):
    """Get the debug_urls, and node/op whitelists for the current run() call.

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`.
      feed_dict: Same as the `feed_dict argument` to `Session.run()`.

    Returns:
      debug_urls: (str or list of str) Debug URLs for the current run() call.
        Currently, the list consists of only one URL that is a file:// URL.
      watch_options: (WatchOptions) The return value of a watch_fn, containing
        options including debug_ops, and whitelists.
    """
    PRINT()

    debug_urls = self.prepare_run_debug_urls(fetches, feed_dict)
    if self._watch_fn is None:
      watch_options = WatchOptions()
    else:
      watch_options = self._watch_fn(fetches, feed_dict)
      if isinstance(watch_options, tuple):
        # For legacy return type (tuples).
        watch_options = WatchOptions(*watch_options)

    return debug_urls, watch_options

  def on_run_end(self, request):
    """See doc of BaseDebugWrapperSession.on_run_end."""
    PRINT()

    return OnRunEndResponse()

  def invoke_node_stepper(self,
                          node_stepper,
                          restore_variable_values_on_exit=True):
    """See doc of BaseDebugWrapperSession.invoke_node_stepper."""

    raise NotImplementedError(
        "NonInteractiveDebugWrapperSession does not support node-stepper mode.")
