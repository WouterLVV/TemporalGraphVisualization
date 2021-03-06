from collections import deque
from typing import List, Dict, Any
import itertools


# --------------------------------------------- #
#              Support classes                  #
# --------------------------------------------- #

class TimeNode:
    """Simple data structure for nodes than can belong to different groups over time"""

    def __init__(self, conns: set = None, max_step: int = -1, id=-1, meta_string=None):
        """Initializes a TimeNode structure

        :param conns: a set of connections this node has in tuple form (neighbour, time step)
        :param max_step: The maximum number of steps in the total graph
        :param id: Node identifier, should be unique over the graph this TimeNode belongs to
        :param meta_string: (Optional) Human readable name of this node
        """
        self.id = id
        self.meta_string = meta_string

        self.nbs = [set() for _ in range(max_step)]
        self.clusters = [None] * max_step  # type: List[TimeCluster or None]

        if conns is not None:
            for (n, t) in conns:
                self.nbs[t].add(n)

    def add_connection(self, n, t : int) -> None:
        """Adds a connection to this node

        :param n: node id of the neighbouring node
        :param t: time step in which this connection exists
        :return: None
        """
        self.nbs[t].add(n)

    def __str__(self):
        return str(self.id) + ((": " + self.meta_string) if self.meta_string is not None else "")


class TimeCluster:
    """Data structure that holds a cluster of connected nodes

    A TimeCluster in practice contains a single connected component at a single time step.
    It should keep track of incoming and outgoing connections and the nodes that belong to each connection
    Each node in members should appear in both the incoming and outgoing values, excepting the first and final layer

    :ivar id: Timecluster identifier, should be unique within the layer
    :ivar incoming: Incoming connections from the previous layer
    :ivar outgoing: Outgoing connections to the following layer
    :ivar members: Set containing all node ids that belong to this cluster
    :ivar layer: The graph layer this cluster belongs to
    """

    def __init__(self, layer: int, id: int):
        """Initializes a TimeCluster object

        :param layer: Layer number this cluster exists in
        :param id: identifier of this cluster, unique within layer
        """
        self.id = id

        self.incoming = dict()
        self.outgoing = dict()
        self.members = set()

        self.insize = self.outsize = 0

        self.layer = layer

    def add(self, node: TimeNode):
        """Adds a single node to this cluster

        :param node: identifier of the node added to this cluster
        :return:
        """

        self.members.add(node)

    def add_connection(self, target, members):
        """Adds a connection from or to the target timecluster for members

        If the connection already exists, the new member(s) is/are added to the existing connection.

        :param TimeCluster target: The TimeCluster to connect to. Should exist in either layer+1 or layer-1.
        :param TimeNode or set members: id or set of ids that belong to this connection.
        :return: None
        """

        t = target.layer
        if type(members) == TimeNode:

            if members not in self.members:
                raise Exception("Connection contains members not in this cluster")

            if t == self.layer + 1:
                if target not in self.outgoing.keys():
                    self.outgoing[target] = set()
                self.outgoing[target].add(members)
            elif t == self.layer - 1:
                if target not in self.incoming.keys():
                    self.incoming[target] = set()
                self.incoming[target].add(members)
            else:
                raise Exception("Connection can only be added to relevant cluster")

        else:
            members = set(members)
            if len(self.members.intersection(members)) != len(members):
                raise Exception("Connection contains members not in this cluster")

            if t == self.layer + 1:
                if target not in self.outgoing.keys():
                    self.outgoing[target] = set()
                self.outgoing[target] = self.outgoing[target].union(members)
            elif t == self.layer - 1:
                if target not in self.incoming.keys():
                    self.incoming[target] = set()
                self.incoming[target] = self.incoming[target].union(members)
            else:
                raise Exception("Connection can only be added to relevant cluster")

    def update(self, min_conn_size):
        """Trim connections according to some minimum connection size

        :param min_conn_size: minimum size of connection to not be removed
        :return: Whether this cluster is alive or is orphaned and has destroyed itself
        """
        self.insize = self.outsize = 0
        for k, v in list(self.incoming.items()):
            if len(v) < min_conn_size:
                del self.incoming[k]
            else:
                self.insize += len(v)

        for k, v in list(self.outgoing.items()):
            if len(v) < min_conn_size:
                del self.outgoing[k]
            else:
                self.outsize += len(v)

        if self.insize + self.outsize == 0:
            self.destroy()
            return False
        return True

    def destroy(self):
        for n in self.members:
            n.clusters[self.layer] = None
        for nb in self.incoming.keys():
            nb.outgoing.pop(self, None)
        for nb in self.outgoing.keys():
            nb.incoming.pop(self, None)

    def __len__(self):
        return len(self.members)

    def __lt__(self, other):
        return len(self) <= len(other)

    def __str__(self):
        return f"TimeCluster: {self.layer}, {self.id}"


# --------------------------------------------- #
#                 Main class                    #
# --------------------------------------------- #

class TimeGraph:
    """Initializes a TimeGraph object

    A Timegraph object holds information about a graph that changes in discrete steps.
    Each step independently contains the full state of the graph in that step and also the clustering in that step.

    :ivar int num_steps: The number of steps in this graph
    :ivar int num_nodes: The number of nodes (not clusters) total
    :ivar nodes: if an int, generates 0-based ids for the nodes, if list or set interpreted as list of ids as used in the data, if dict interpreted as {node_id: metadata}
    :ivar clusters: List of list of clusters
    """
    def __init__(self, conns, nodes: int or List[str] or Dict[Any, str] or set[Any], num_steps: int, minimum_cluster_size=1, minimum_connection_size=1):
        """Initializes a TimeGraph

        :param conns: Iterable of undirected connections in the form of (head, tail, timestep) or in dictionary format
                        by form of {timestamp: [(head, tail), ...]} or as a list of lists where the index of the list is the timestep.
        :param nodes: Either an int (#nodes) or a list/set [ node_id, ... ] or a dict {node_id: node_metadata}
        :param num_steps: Total number of steps
        """

        self.num_steps = num_steps
        self.num_nodes = nodes

        self.min_clust_size = minimum_cluster_size
        self.min_conn_size = minimum_connection_size


        if isinstance(conns, dict):  # {timestamp: [iterable of connections in timestamp]}
            if self.num_steps < 0:   # If number of steps is not given, derive from data
                self.num_steps = max(conns.keys()) + 1  # timestamps should be 0-indexed.
            if nodes is None:        # If no node ids were given, derive from data
                nodes = set([x for pair in conns.values() for x in pair])

            self._make_nodes(nodes)

            for t, l in conns.items():
                for (f, b) in l:
                    self.nodes[f].add_connection(self.nodes[b], t)
                    self.nodes[b].add_connection(self.nodes[f], t)

        elif isinstance(conns, list):  # Either a list of [(a, b, t), ...] or [[(a,b), ...], [(b,c), ...]]
            list_of_list_of_pairs = False
            try:
                if len(conns[0][0]) == 2:
                    list_of_list_of_pairs = True  # If true, that interprets the index of a list as the timestamp
            except TypeError:
                pass

            if list_of_list_of_pairs:  # List of list of connections
                if self.num_steps < 0:
                    self.num_steps = len(conns)
                if nodes is None:
                    nodes = set([x for step in conns for pair in step for x in pair])
                self._make_nodes(nodes)

                for t, l in enumerate(conns):
                    for (f, b) in l:
                        self.nodes[f].add_connection(self.nodes[b], t)
                        self.nodes[b].add_connection(self.nodes[f], t)

            else:  # List of triplets of (a, b, t)
                if self.num_steps < 0:
                    self.num_steps = max([x[2] for x in conns]) + 1
                if nodes is None:
                    nodes = set([x for (a, b, _) in conns for x in (a, b)])
                self._make_nodes(nodes)

                for (f, b, t) in conns:
                    self.nodes[f].add_connection(self.nodes[b], t)
                    self.nodes[b].add_connection(self.nodes[f], t)

        else:  # some iterable with triplets (a, b, t)
            if self.num_steps < 0:
                self.num_steps = max([x[2] for x in conns]) + 1  # Find largest timestamp value
            if nodes is None:
                nodes = set([x for (a, b, _) in conns for x in (a, b)])
            self._make_nodes(nodes)

            for (f, b, t) in conns:
                self.nodes[f].add_connection(self.nodes[b], t)
                self.nodes[b].add_connection(self.nodes[f], t)

        self.layers = [self.create_layer_components(t) for t in range(self.num_steps)]
        self.connect_clusters()
        self.layers = [
            [cluster for cluster in layer if cluster.update(self.min_conn_size)]
            for layer in self.layers
        ]

        self.clusters = [x for t in range(self.num_steps) for x in self.layers[t]]


    def _make_nodes(self, nodes):
        num_steps = self.num_steps
        if isinstance(nodes, int):  # Create n nodes with 0-based index id.
            self.nodes = {d: TimeNode(None, num_steps, d) for d in range(nodes)}
        elif isinstance(nodes, list) or isinstance(nodes, set):  # Take node ids from
            self.nodes = {d: TimeNode(max_step=num_steps, id=d) for d in nodes}
        elif isinstance(nodes, dict):
            self.nodes = {node_id: TimeNode(max_step=num_steps, id=node_id, meta_string=node_meta) for node_id, node_meta in nodes.items()}

    def get_cluster(self, t, id):
        return self.layers[t][id]

    def create_layer_components(self, t: int):
        """Function that creates connected components in each layer

        For a layer the connected components are built by progressively flood-filling any node that is not yet in a cluster

        :param t: Time step of this layer
        :return: List of clusters in this layer
        """
        clusts = []
        seen = set()
        ctr = 0
        for node in self.nodes.values():
            if node.id in seen:
                continue

            seen.add(node.id)
            clust = TimeCluster(t, ctr)

            # Populate cluster through flood fill
            q = deque()
            q.append(node)
            while len(q) > 0:
                n = q.pop()

                clust.add(n)
                n.clusters[t] = clust
                for nb in n.nbs[t]:
                    if nb.id not in seen:
                        q.append(nb)
                        seen.add(nb.id)

            if len(clust) >= self.min_clust_size:
                clusts.append(clust)
                ctr += 1
            else:
                clust.destroy()
        return clusts

    def connect_clusters(self):
        """Function that takes the clustering per layer and interconnects clusters

        A connection between two clusters is made if they are in adjacent layers and share the same node as member.

        :return: None
        """

        for n in self.nodes.values():

            for t in range(self.num_steps - 1):
                head = n.clusters[t]
                tail = n.clusters[t + 1]

                if head is None or tail is None:
                    continue

                head.add_connection(tail, n)
                tail.add_connection(head, n)

    # Global (time graph) statistics

    def density(self):
        """ How much data would be visualised in this temporal graph, relative to the graph dimensions?
        """

        presence_count = 0

        for cluster in self.clusters:
            if len(cluster.members) >= self.min_clust_size:
                presence_count += max(cluster.insize, cluster.outsize)

        if len(self.nodes) * len(self.layers) > 0:
            return presence_count / len(self.nodes) / len(self.layers)
        return 0

    def num_clusters(self):
        return len(self.clusters)

    def avg_num_clusters_per_time_step(self):
        return len(self.clusters) / self.num_steps

    def num_events(self):
        """ Counts events: splits, merges, starts, ends, stables.
        """

        start_count, end_count, merge_count, split_count, stable_count = 0, 0, 0, 0, 0

        for cluster in self.clusters:
            if max(cluster.insize, cluster.outsize) < self.min_clust_size:
                continue

            # count starts and merges
            num_incoming = 0
            for k, v in cluster.incoming.items():
                if len(k) < self.min_clust_size or len(v) < self.min_conn_size:
                    continue
                num_incoming += 1
            if num_incoming == 0:
                start_count += 1
            else:
                merge_count += num_incoming - 1

            # count ends and splits
            num_outgoing = 0
            for k, v in cluster.outgoing.items():
                if len(k) < self.min_clust_size or len(v) < self.min_conn_size:
                    continue
                num_outgoing += 1
            if num_outgoing == 0:
                end_count += 1
            else:
                split_count += num_outgoing - 1

            # count stables
            if num_incoming == 1 and num_outgoing == 1:
                stable_count += 1

        return start_count, end_count, merge_count, split_count, stable_count

    def num_events_per_time_step(self):
        return self.num_events() / self.num_steps
    
    def average_relative_continuity(self):
        return sum(self.relative_continuity()) / self.num_steps
    
    def average_absolute_continuity(self):
        return sum(self.absolute_continuity()) / self.num_steps

    def average_relative_continuity_diff(self):
        return sum(self.relative_continuity_diff()) / self.num_steps

    def average_absolute_continuity_diff(self):
        return sum(self.absolute_continuity_diff()) / self.num_steps

    def normalized_absolute_continuity(self):
        return sum(self.absolute_continuity()) / sum(self.layer_num_members())

    def normalized_absolute_continuity_diff(self):
        return sum(self.absolute_continuity_diff()) / sum(self.layer_num_members())

    # Local (time step) statistics

    def layer_in_out_diff(self):
        return [abs(sum(map(lambda x: x.insize, layer)) - sum(map(lambda x: x.outsize, layer))) for layer in self.layers]

    def layer_num_clusters(self):
        return list(map(len, self.layers))

    def layer_num_members(self):
        return list(map(sum, map(lambda x: map(len, x), self.layers)))

    def relative_continuity(self):
        res = []
        for layer in self.layers:
            if len(layer) == 0:
                res.append(0.)
            else:
                res.append(sum(map(lambda c: max(map(len, c.outgoing.values()), default=1), layer))/sum(map(len, layer)))
        return res
    
    def absolute_continuity(self):
        res = []
        for layer in self.layers:
            if len(layer) == 0:
                res.append(0.)
            else:
                res.append(sum(map(lambda c: max(map(len, c.outgoing.values()), default=1), layer)))
        return res

    def relative_continuity_diff(self):
        hom = self.relative_continuity()
        res = [0.]
        for i in range(len(hom)-1):
            res.append(abs(hom[i] - hom[i+1]))
        return res
    
    def absolute_continuity_diff(self):
        con = self.absolute_continuity()
        res = [0.]
        for i in range(len(con) - 1):
            res.append(abs(con[i] - con[i + 1]))
        return res
