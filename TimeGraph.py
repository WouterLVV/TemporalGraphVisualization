from collections import deque
from typing import List


# --------------------------------------------- #
#              Support classes                  #
# --------------------------------------------- #

class TimeNode:
    """Simple data structure for nodes than can belong to different groups over time"""

    def __init__(self, conns: set = None, max_step: int = -1, id=-1, name=None):
        """Initializes a TimeNode structure

        :param conns: a set of connections this node has in tuple form (neighbour, time step)
        :param max_step: The maximum number of steps in the total graph
        :param id: Node identifier, should be unique over the graph this TimeNode belongs to
        :param name: (Optional) Human readable name of this node
        """
        self.id = id
        self.name = name

        self.nbs = [set() for _ in range(max_step)]
        self.clusters = [None] * max_step  # type: List[TimeCluster]

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
    :ivar nodes: List of nodes
    :ivar clusters: List of list of clusters
    """
    def __init__(self, conns, nodes: int or List[str], num_steps: int):
        """Initializes a TimeGraph

        :param conns: Iterable of undirected connections in the form of (head, tail, timestep)
        :param nodes: Either an int (#nodes) or a list [ node_name, ... ]
        :param num_steps: Total number of steps
        """

        self.num_steps = num_steps
        self.num_nodes = nodes

        if isinstance(nodes, int):
            self.nodes = [TimeNode(None, num_steps, id) for id in range(nodes)]
        else:
            self.nodes = [TimeNode(max_step=num_steps, id=i, name=d) for i, d in enumerate(nodes)]


        for (f, b, t) in conns:
            self.nodes[f].add_connection(self.nodes[b], t)
            self.nodes[b].add_connection(self.nodes[f], t)

        self.layers = [self.create_layer_components(t) for t in range(self.num_steps)]
        self.clusters = [x for t in range(self.num_steps) for x in self.layers[t]]
        self.connect_clusters()

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
        for node in self.nodes:
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

            clusts.append(clust)
            ctr += 1
        return clusts

    def connect_clusters(self):
        """Function that takes the clustering per layer and interconnects clusters

        A connection between two clusters is made if they are in adjacent layers and share the same node as member.

        :return: None
        """

        for n in self.nodes:

            for t in range(self.num_steps - 1):
                head = n.clusters[t]
                tail = n.clusters[t + 1]

                head.add_connection(tail, n)
                tail.add_connection(head, n)

    def average_neighbours(self, minimum_cluster_size=1, minimum_connection_size=1):
        num_connections = 0
        num_clusters = 0
        for cluster in self.clusters:
            if len(cluster) < minimum_cluster_size:
                continue
            cluster_connections = 0
            for k, v in cluster.outgoing.items():
                if len(k) < minimum_cluster_size or len(v) < minimum_connection_size:
                    continue
                cluster_connections += 1
            for k, v in cluster.incoming.items():
                if len(k) < minimum_cluster_size or len(v) < minimum_connection_size:
                    continue
                cluster_connections += 1
            if cluster_connections > 0:
                num_clusters += 1
                num_connections += cluster_connections

        return num_connections / num_clusters
