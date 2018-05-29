"""Data structures and algorithms for profiling information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..examples import debug_v
def PRINT(txt=""):
    debug_v.PRINT(txt)


class ProfileDatum(object):
  """Profile data point."""

  def __init__(self,
               device_name,
               node_exec_stats,
               file_path,
               line_number,
               func_name,
               op_type):
    """Constructor.

    Args:
      device_name: (string) name of the device.
      node_exec_stats: `NodeExecStats` proto.
      file_path: path to the source file involved in creating the op.
      line_number: line number in the file involved in creating the op.
      func_name: name of the function that the line belongs to.
      op_type: (string) Operation type.
    """
    PRINT()
    self.device_name = device_name
    self.node_exec_stats = node_exec_stats
    self.file_path = file_path
    self.line_number = line_number
    self.func_name = func_name
    if self.file_path:
      self.file_line_func = "%s:%d(%s)" % (
          os.path.basename(self.file_path), self.line_number, self.func_name)
    else:
      self.file_line_func = ""
    self.op_type = op_type
    self.start_time = self.node_exec_stats.all_start_micros
    self.op_time = (self.node_exec_stats.op_end_rel_micros -
                    self.node_exec_stats.op_start_rel_micros)

  @property
  def exec_time(self):
    """Op execution time plus pre- and post-processing."""
    PRINT()
    return self.node_exec_stats.all_end_rel_micros


class AggregateProfile(object):
  """Profile summary data for aggregating a number of ProfileDatum."""

  def __init__(self, profile_datum):
    """Constructor.

    Args:
      profile_datum: (`ProfileDatum`) an instance of `ProfileDatum` to
        initialize this object with.
    """
    PRINT()

    self.total_op_time = profile_datum.op_time
    self.total_exec_time = profile_datum.exec_time
    device_and_node = "%s:%s" % (profile_datum.device_name,
                                 profile_datum.node_exec_stats.node_name)
    self._node_to_exec_count = {device_and_node: 1}

  def add(self, profile_datum):
    """Accumulate a new instance of ProfileDatum.

    Args:
      profile_datum: (`ProfileDatum`) an instance of `ProfileDatum` to
        accumulate to this object.
    """
    PRINT()

    self.total_op_time += profile_datum.op_time
    self.total_exec_time += profile_datum.exec_time
    device_and_node = "%s:%s" % (profile_datum.device_name,
                                 profile_datum.node_exec_stats.node_name)

    device_and_node = "%s:%s" % (profile_datum.device_name,
                                 profile_datum.node_exec_stats.node_name)
    if device_and_node in self._node_to_exec_count:
      self._node_to_exec_count[device_and_node] += 1
    else:
      self._node_to_exec_count[device_and_node] = 1

  @property
  def node_count(self):
    PRINT()
    return len(self._node_to_exec_count)

  @property
  def node_exec_count(self):
    PRINT()
    return sum(self._node_to_exec_count.values())
