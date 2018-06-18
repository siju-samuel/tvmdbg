"""Classes and functions that help to inspect Python source w.r.t. TVM graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re

import numpy as np

from tvm.tools.debug.util import profiling

_TVM_BASEDIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.normpath(os.path.abspath(__file__))))))

UNCOMPILED_SOURCE_SUFFIXES = (".py")
COMPILED_SOURCE_SUFFIXES = (".pyc", ".pyo")


def _norm_abs_path(file_path):
    return os.path.normpath(os.path.abspath(file_path))


def _is_extension_uncompiled(file_path):
    _, extension = os.path.splitext(file_path)
    return extension.lower() in UNCOMPILED_SOURCE_SUFFIXES


def _is_extension_compiled(file_path):
    _, extension = os.path.splitext(file_path)
    return extension.lower() in COMPILED_SOURCE_SUFFIXES


def _cvrt_watch_key_to_tensor(watch_key):
    return watch_key[:watch_key.rfind(":")]


def guess_is_tvm_py_library(py_file_path):
    """Guess whether a Python source file is a part of the TVM library.

    Special cases:
      1) Returns False for unit-test files in the library (*_test.py),
      2) Returns False for files under python/debug/examples.

    Args:
      py_file_path: full path of the Python source file in question.

    Returns:
      (`bool`) Whether the file is a part of the TVM library.

    Raises:
      ValueError: if the extension name of py_file_path does not indicate a Python
        source file (compiled or uncomplied).
    """
    if (not _is_extension_uncompiled(py_file_path) and
            not _is_extension_compiled(py_file_path)):
        raise ValueError(
            "Input file path (%s) is not a Python source file." % py_file_path)
    py_file_path = _norm_abs_path(py_file_path)

    return (py_file_path.startswith(_TVM_BASEDIR) and
            not py_file_path.endswith("_test.py") and
            not os.path.dirname(py_file_path).endswith(
                os.path.normpath("python/debug/examples")))


def load_source(source_file_path):
    """Load the Python source file with printable format lines.

    Args:
      source_file_path: (`str`) path to the source file.

    Returns:
      1. Content of source file splitted by line.
      2. Source content line width
    """
    with open(source_file_path, "rU") as fopen:
        source_text = fopen.read()
    source_lines = source_text.split("\n")
    line_num_width = int(np.ceil(np.log10(len(source_lines)))) + 3
    return source_lines, line_num_width


def annotate_source(dump,
                    source_file_path,
                    do_dumped_tensors=False,
                    file_stack_top=False,
                    min_line=None,
                    max_line=None):
    """Annotate a Python source file with a list of ops created at each line.

    (The annotation doesn't change the source file itself.)

    Args:
      dump: (`DebugDumpDir`) A `DebugDumpDir` object of which the Python graph
        has been loaded.
      source_file_path: (`str`) Path to the source file being annotated.
      do_dumped_tensors: (`str`) Whether dumped Tensors, instead of ops are to be
        used to annotate the source file.
      file_stack_top: (`bool`) Whether only the top stack trace in the
        specified source file is to be annotated.
      min_line: (`None` or `int`) The 1-based line to start annotate the source
        file from (inclusive).
      max_line: (`None` or `int`) The 1-based line number to end the annotation
        at (exclusive).

    Returns:
      A `dict` mapping 1-based line number to a list of op name(s) created at
        that line, or tensor names if `do_dumped_tensors` is True.

    Raises:
      ValueError: If the dump object does not have a Python graph set.
    """

    py_graph = dump.python_graph
    if not py_graph:
        raise ValueError("Cannot perform source annotation due to a lack of set "
                         "Python graph in the dump object")

    source_file_path = _norm_abs_path(source_file_path)

    line_to_op_names = {}
    for op in py_graph.get_operations():
        for file_path, line_number, _, _ in reversed(dump.node_traceback(op.name)):
            if (min_line is not None and line_number < min_line or
                    max_line is not None and line_number >= max_line):
                continue

            if _norm_abs_path(file_path) != source_file_path:
                continue

            if do_dumped_tensors:
                watch_keys = dump.debug_watch_keys(op.name)
                # Convert watch keys to unique Tensor names.
                items_to_append = list(
                    set(map(_cvrt_watch_key_to_tensor, watch_keys)))
            else:
                items_to_append = [op.name]

            if line_number in line_to_op_names:
                line_to_op_names[line_number].extend(items_to_append)
            else:
                line_to_op_names[line_number] = items_to_append

            if file_stack_top:
                break

    return line_to_op_names


def list_source_files_against_dump(dump,
                                   path_regex_whitelist=None,
                                   node_name_regex_whitelist=None):
    """Generate a list of source files with information regarding ops and tensors.

    Args:
      dump: (`DebugDumpDir`) A `DebugDumpDir` object of which the Python graph
        has been loaded.
      path_regex_whitelist: A regular-expression filter for source file path.
      node_name_regex_whitelist: A regular-expression filter for node names.

    Returns:
      A list of tuples regarding the Python source files involved in constructing
      the ops and tensors contained in `dump`. Each tuple is:
        (source_file_path, is_tvm_library, num_nodes, num_tensors, num_dumps,
         first_line)

        is_tvm_library: (`bool`) A guess of whether the file belongs to the
          TVM Python library.
        num_nodes: How many nodes were created by lines of this source file.
          These include nodes with dumps and those without.
        num_tensors: How many Tensors were created by lines of this source file.
          These include Tensors with dumps and those without.
        num_dumps: How many debug Tensor dumps were from nodes (and Tensors)
          that were created by this source file.
        first_line: The first line number (1-based) that created any nodes or
          Tensors in this source file.

      The list is sorted by ascending order of source_file_path.

    Raises:
      ValueError: If the dump object does not have a Python graph set.
    """

    py_graph = dump.python_graph
    if not py_graph:
        raise ValueError("Cannot generate source list due to a lack of set "
                         "Python graph in the dump object")

    path_to_node_names = collections.defaultdict(set)
    path_to_tensor_names = collections.defaultdict(set)
    path_to_first_line = {}
    tensor_name_to_num_dumps = {}

    path_regex = (re.compile(path_regex_whitelist)
                  if path_regex_whitelist else None)
    node_name_regex = (re.compile(node_name_regex_whitelist)
                       if node_name_regex_whitelist else None)

    to_skip_file_paths = set()
    for op in py_graph.get_operations():
        if node_name_regex and not node_name_regex.match(op.name):
            continue

        for file_path, line_number, _, _ in dump.node_traceback(op.name):
            file_path = _norm_abs_path(file_path)
            if (file_path in to_skip_file_paths or
                    path_regex and not path_regex.match(file_path) or
                    not os.path.isfile(file_path)):
                to_skip_file_paths.add(file_path)
                continue

            path_to_node_names[file_path].add(op.name)
            if file_path in path_to_first_line:
                if path_to_first_line[file_path] > line_number:
                    path_to_first_line[file_path] = line_number
            else:
                path_to_first_line[file_path] = line_number

            for output_tensor in op.outputs:
                tensor_name = output_tensor.name
                path_to_tensor_names[file_path].add(tensor_name)

            watch_keys = dump.debug_watch_keys(op.name)
            for watch_key in watch_keys:
                node_name, output_slot, debug_op = watch_key.split(":")
                tensor_name = "%s:%s" % (node_name, output_slot)
                if tensor_name not in tensor_name_to_num_dumps:
                    tensor_name_to_num_dumps[tensor_name] = len(
                        dump.get_tensors(node_name, int(output_slot), debug_op))

    path_to_num_dumps = {}
    for path in path_to_tensor_names:
        path_to_num_dumps[path] = sum(
            tensor_name_to_num_dumps.get(tensor_name, 0)
            for tensor_name in path_to_tensor_names[path])

    output = []
    for file_path in path_to_node_names:
        output.append((
            file_path,
            guess_is_tvm_py_library(file_path),
            len(path_to_node_names.get(file_path, {})),
            len(path_to_tensor_names.get(file_path, {})),
            path_to_num_dumps.get(file_path, 0),
            path_to_first_line[file_path]))

    return sorted(output, key=lambda x: x[0])
