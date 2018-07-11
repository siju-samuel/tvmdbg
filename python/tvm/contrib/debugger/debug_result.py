"""Graph debug results dumping class."""
import os
import json
import numpy as np

class DebugResult():
    """Graph debug data module.

    Data dump module manage all the debug data formatting.
    Output data and input graphs are formatted and the dump to files.
    Frontend read these data and graph for visualization.

    Parameters
    ----------
    nodes_list : list
        List of all the nodes presented in the graph

    node_stats : list
        Memory buffer contain each node's output data.

    dump_path : str
        Output data path is read/provided from frontend
    """
    def __init__(self, nodes_list, node_stats, dump_path):
        self._nodes_list = nodes_list
        self._dump_path = dump_path
        self._out_stats = node_stats
        self._time_list = []

    def get_nodes_list(self):
        """Return the nodes list
        """
        return self._nodes_list

    def set_time(self, time):
        """set the timestamp to the timelist, the list is appended in the same order as node list

        Parameters
        ----------
        time : float
            The time for a particular operation, added to the list
        """
        self._time_list.append(time)

    def dump_output_tensor(self):
        """Dump the outputs to a temporary folder

        Dump path is read from frontend.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        #cleanup existing tensors before dumping
        self._cleanup_tensors()
        eid = 0
        order = 0
        for node, time in zip(self._nodes_list, self._time_list):
            num_outputs = 1 if node['op'] == 'param' \
                            else int(node['attrs']['num_outputs'])
            for j in range(num_outputs):
                ndbuffer = self._out_stats[eid]
                eid += 1
                order += time
                key = node['name'] + "_" + str(j) + "__" + str(order) + ".npy"
                dump_file = str(self._dump_path + key.replace("/", "_"))
                np.save(dump_file, ndbuffer.asnumpy())
                os.rename(dump_file, dump_file.rpartition('.')[0])

    def dump_graph_json(self, graph):
        """Dump json formatted graph.

        Parameters
        ----------
        graph : json format
            json formatted NNVM graph contain list of each node's
            name, shape and type.

        Returns
        -------
        none
        """
        graph_dump_file_name = '_tvmdbg_graph_dump.json'
        with open((self._dump_path + graph_dump_file_name), 'w') as outfile:
            json.dump(graph, outfile, indent=2, sort_keys=False)

    def _cleanup_tensors(self):
        """Remove the files other than graph in the dump path
        """
        for filename in os.listdir(self._dump_path):
            if os.path.isfile(filename) and not filename.endswith(".json"):
                os.remove(filename)
