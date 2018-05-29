"""Common values and methods for TVM Debugger."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..examples import debug_v
def PRINT(txt=""):
    debug_v.PRINT(txt)

def get_graph_element_name(elem):
  """Obtain the name or string representation of a graph element.

  If the graph element has the attribute "name", return name. Otherwise, return
  a __str__ representation of the graph element. Certain graph elements, such as
  `SparseTensor`s, do not have the attribute "name".

  Args:
    elem: The graph element in question.

  Returns:
    If the attribute 'name' is available, return the name. Otherwise, return
    str(fetch).
  """
  if hasattr(elem, "attr"):
    val = elem.attr("name")
  else:
    val = elem.name if hasattr(elem, "name") else str(elem)
  PRINT(val)
  return val


def get_flattened_names(feeds_or_fetches):
  """Get a flattened list of the names in run() call feeds or fetches.

  Args:
    feeds_or_fetches: Feeds or fetches of the `Session.run()` call. It maybe
      a Tensor, an Operation or a Variable. It may also be nested lists, tuples
      or dicts. See doc of `Session.run()` for more details.

  Returns:
    (list of str) A flattened list of fetch names from `feeds_or_fetches`.
  """
  PRINT()

  lines = []
  if isinstance(feeds_or_fetches, (list, tuple)):
    for item in feeds_or_fetches:
      lines.extend(get_flattened_names(item))
  elif isinstance(feeds_or_fetches, dict):
    for key in feeds_or_fetches:
      lines.extend(get_flattened_names(feeds_or_fetches[key]))
  elif ';' in feeds_or_fetches:
    names = feeds_or_fetches.split(";")
    for name in names:
      if name:
        lines.extend(get_flattened_names(name))
  else:
    lines.append(get_graph_element_name(feeds_or_fetches))
  return lines

#def get_flattened_names(feeds_or_fetches):
#  """Get a flattened list of the names in run() call feeds or fetches.
#
#  Args:
#    feeds_or_fetches: Feeds or fetches of the `Session.run()` call. It maybe
#      a Tensor, an Operation or a Variable. It may also be nested lists, tuples
#      or dicts. See doc of `Session.run()` for more details.

#  Returns:
#    (list of str) A flattened list of fetch names from `feeds_or_fetches`.
#  """
#  PRINT()

#  lines = []
#  if isinstance(feeds_or_fetches, (list, tuple)):
#    for item in feeds_or_fetches:
#      lines.extend(get_flattened_names(item))
#  elif isinstance(feeds_or_fetches, dict):
#    for key in feeds_or_fetches:
#      lines.extend(get_flattened_names(feeds_or_fetches[key]))
#  else:
#    # This ought to be a Tensor, an Operation or a Variable, for which the name
#    # attribute should be available. (Bottom-out condition of the recursion.)
#    lines.append(get_graph_element_name(feeds_or_fetches))

#  for line in lines:
#    PRINT(line)
#  return lines