from __future__ import annotations

import math
from collections import Counter, deque
from typing import List, Set, Dict, Optional, Tuple

import cairocffi as cairo

from TimeGraph import TimeGraph, TimeCluster
from drawing_utils import coloured_bezier

# from sklearn.preprocessing import normalize
# import cairo

INCOMING = FORWARD = True
OUTGOING = BACKWARD = False


class NotRootException(Exception):
    pass


class UnorderedException(Exception):
    pass


class NotEndpointException(Exception):
    pass


class SugiyamaCluster:
    def __init__(self, tc: TimeCluster, height_method):
        # Link to original TimeCluster and vice versa
        self.tc = tc
        tc.sc = self

        # Connection properties
        self.incoming = dict()  # Filled by build(), k is SugiyamaCluster, v is set of TimeNodes
        self.outgoing = dict()  # Filled by build(), k is SugiyamaCluster, v is set of TimeNodes
        self.neighbours = dict()  # Filled by build(), k is SugiyamaCluster, v is set of TimeNodes

        self.insize = 0  # Filled by build(), number of incoming connections
        self.outsize = 0  # Filled by build(), number of outgoing connections

        self.largest_incoming = 0  # Filled by build(), size of the largest incoming connection
        self.largest_outgoing = 0  # Filled by build(), size of the largest outgoing connection

        self.members = self.tc.members  # Set of TimeNodes in this cluster

        # Order properties
        self.rank = -1
        self.inrank = -1
        self.outrank = -1
        self.wanted_direction = 0

        # Alignment properties
        self.root = self
        self.align = self
        self.chain_length = 1

        # Location properties
        self.x = -1
        self.y = -1
        self._y = -1  # storage value if self.y should not be changed immediately

        # Drawing properties
        self.draw_size = self.draw_height(height_method)

    def draw_height(self, method: str) -> float:
        """Determine vertical size of this cluster depending on its size

        :param method: The function to apply to the member size. Accepts 'linear', 'sqrt', 'log' or 'constant'
        :return: Size of this cluster
        """
        if method == 'linear':
            return len(self.members)
        elif method == 'sqrt':
            return math.sqrt(len(self.members))
        elif method == 'log':
            return math.log(len(self.members))
        elif method == 'constant':
            return 1.
        else:
            return 0.

    def build(self) -> None:
        """Builds the neighbour set and related data from the base TimeCluster

        """
        for c, connection_nodes in self.tc.incoming.items():
            self.incoming[c.sc] = connection_nodes

        for c, connection_nodes in self.tc.outgoing.items():
            self.outgoing[c.sc] = connection_nodes

        self.insize = self.tc.insize
        self.outsize = self.tc.outsize
        self.neighbours = {**self.incoming, **self.outgoing}
        self.largest_incoming = max(map(len, self.incoming.values()), default=0)
        self.largest_outgoing = max(map(len, self.outgoing.values()), default=0)

    def update_cluster_ranks(self) -> Tuple[float, float]:
        """Update the rank information of this cluster

        Rank information is determined by the average rank of the incoming and outgoing connections. Since ranks can
        vary wildly between layers, inrank and outrank can not be compared to each other or this clusters own rank.

        :return: The average incoming rank and the average outgoing rank, or -1 for either if it has none on that side
        """
        if self.insize > 0:
            inrank = sum([nb.rank * len(conn) for nb, conn in self.incoming.items()]) / self.insize
        else:
            inrank = -1

        if self.outsize > 0:
            outrank = sum([n.rank * len(l) for n, l in self.outgoing.items()]) / self.outsize
        else:
            outrank = -1

        self.inrank = inrank
        self.outrank = outrank
        return inrank, outrank

    def update_cluster_ranks_median(self) -> Tuple[float, float]:
        """Update the rank information of this cluster using median instead of average

        Rank information is determined by the median rank of the incoming and outgoing connections. Since ranks can
        vary wildly between layers, inrank and outrank can not be compared to each other or this clusters own rank.

        :return: The median incoming rank and the median outgoing rank
        """
        inc = list(self.incoming.items())
        out = list(self.outgoing.items())

        self.inrank, self.outrank = self.weighted_median_rank(inc), self.weighted_median_rank(out)
        return self.inrank, self.outrank

    @staticmethod
    def weighted_median_rank(nbs: List[(SugiyamaCluster, Set)]) -> float:
        """Of a list of neighbours, return the median ranked neighbour adjusted for integer weighted connections

        Instead of taking the median connection, this function finds the connection that the median ranked node belongs to
        :param nbs: list of neighbours and connection members
        :return: The rank of the cluster of which the median node is connected
        """
        if len(nbs) == 0:
            return -1

        nbs.sort(key=lambda x: x[0].rank)
        inc_total = sum(map(lambda x: len(x[1]), nbs))
        ptr = 0
        inc_sum = 0.
        while inc_sum < inc_total / 2.:
            inc_sum += len(nbs[ptr][1])
            ptr += 1
        if inc_total % 2 == 0 and inc_sum == inc_total // 2:
            med = (nbs[ptr][0].rank + nbs[ptr - 1][0].rank) / 2.
        else:
            med = nbs[ptr - 1][0].rank
        return med

    def reset_alignment(self) -> None:
        """
        Reset the values used in determining alignment to their default values
        """
        self.root = self
        self.align = self
        self.chain_length = 1

    def reset_endpoint(self) -> None:
        """
        If this cluster is the second to last of a chain, make it the last in the chain
        The cluster removed from the chain should be reset separately
        """
        if self.align == self.root or self.align.align != self.align.root:
            return
        self.align = self.root
        self.root.chain_length = max(1, self.root.chain_length - 1)

    def largest_median_connection(self, lower=True, direction=INCOMING) -> (SugiyamaCluster, int):
        """Returns the cluster with the largest connection to this one.

        If multiple candidates with equal connection weight exist, returns the lower median in ordering

        :param self: Cluster to find the median median of
        :param lower: Flag to take either the upper or lower median if even amount
        :param direction: Flag for direction. True is incoming, False is outgoing
        :return: The cluster and the weight of the connection
        """

        if direction:  # is INCOMING
            connections = list(self.incoming.items())
        else:  # is OUTGOING
            connections = list(self.outgoing.items())

        if len(connections) == 0:
            return None, 0

        connections.sort(key=lambda x: (len(x[1]), x[0].rank), reverse=True)
        ptr = 0
        while ptr < len(connections) and len(connections[ptr][1]) == len(connections[0][1]):
            ptr += 1

        brother = connections[(ptr - (1 if lower else 0)) // 2][0]
        connsize = len(connections[0][1])
        return brother, connsize

    def align_with(self, next_cluster: SugiyamaCluster) -> None:
        """puts next_cluster as the next link in this chain, if self is an endpoint

        :param next_cluster: The cluster to align with current chain
        """
        if self.align != self.root:
            raise NotEndpointException(f"Can only align if self is an endpoint, but {self} is aligned with {self.align}")
        self.align = next_cluster
        next_cluster.root = self.root
        next_cluster.align = self.root
        self.root.chain_length += 1

    def update_wanted_direction(self) -> int:
        """Function to calculate which direction an alignment would like to move in depending on the ranks of its connections

        :return: The direction (positive is downwards, negative is upwards) and strength of the direction
        """
        if self.root != self:
            raise NotRootException("You can only call wanted_direction on an alignment root.")

        # l holds the alignment for quick access in all directions
        l = []

        cluster = self
        while True:
            l.append(cluster)
            if cluster.align == self:
                break
            cluster = cluster.align

        total = 0
        for i in range(1, len(l)):
            cluster = l[i]
            # Compare incoming connections to the rank of the previous in the alignment
            for k, v in l[i].incoming.items():
                total += len(v) * (k.rank - l[i - 1].rank)

        for i in range(0, len(l) - 1):
            # Compare outgoing connections to the rank of the next in the alignment
            for k, v in l[i].outgoing.items():
                total += len(v) * (k.rank - l[i + 1].rank)

        # The first and last elements of l are not aligned further for a reason
        # So we have to factor in the direction of where this alignment would have liked to go
        left, _ = l[0].largest_median_connection(direction=INCOMING)
        if left is not None and left.align.tc.layer == l[0].tc.layer:
            total += left.align.rank - l[0].rank

        right, _ = l[-1].largest_median_connection(direction=OUTGOING)
        if right is not None and right.root.tc.layer <= l[-1].tc.layer:
            prev = right.root
            while prev.align != right:
                prev = prev.align
            total += prev.rank - l[-1].rank

        self.wanted_direction = total
        return total

    def pos(self) -> Tuple[float, float]:
        """Returns the current x and y coordinate of this cluster. This should be treated as the center of this cluster.

        :return: tuple of x and y coordinate
        """
        return self.x, self.y

    def __str__(self):
        return f"SugiyamaCluster {str((self.tc.layer, self.tc.id))}/{self.rank} at {self.y}"

    def __len__(self):
        return len(self.members)


class SugiyamaLayout:

    ####################################################################################################################
    # -------------------------------------------- init Functions ---------------------------------------------------- #
    ####################################################################################################################

    def __init__(self, g: TimeGraph,
                 line_width=-1., line_spacing=0.0,
                 line_curviness=0.3,
                 horizontal_density=1., vertical_density=1.,
                 cluster_width=-1,
                 cluster_height_method='linear',
                 font_size=-1,
                 verbose=False):

        # Set basic information from the time graph
        self.g = g
        self.num_layers = self.g.num_steps

        # Set different collection objects
        self.clusters, self.layers = self.build_clusters(cluster_height_method)
        self.ordered = []  # Filled by reset_order()
        self.reset_order()

        # Set flags
        self.is_ordered = False
        self.is_aligned = False
        self.is_located = False

        # General info
        max_cluster = max(self.clusters, key=len)
        self.max_cluster_height = max_cluster.draw_size
        self.max_cluster_size = len(max_cluster)  # Amount of elements in cluster
        self.max_bundle_size = max([max(map(len, cluster.neighbours.values())) for cluster in self.clusters])  # Amount elements in connection
        self.max_num_connection = max(map(lambda x: len(x.incoming), self.clusters))  # Amount of connections on one side of a cluster

        # Location settings
        # 1 point = 0.352 mm, or 3 points = 1 mm
        self.xseparation_frac = horizontal_density  # (fraction) from user
        self.yseparation_frac = vertical_density    # (fraction) from user
        self.line_spacing = line_spacing            # (fraction) from user; relative to line_width
        self.line_width = line_width if line_width >= 0. else self.auto_line_width()  # (in points) from user
        self.line_curviness = line_curviness                                          # (fraction) from user; (see curve_offset)

        self.scale = 1.  # (fraction) Scale the image

        # automatically set some drawing parameters based on the graph data
        self.yseparation = self.max_bundle_size * self.line_width * self.yseparation_frac  # (in points)
        self.xseparation = self.max_bundle_size * self.line_width * self.xseparation_frac  # (in points)
        self.ymargin = self.yseparation  # (in points)
        self.xmargin = self.xseparation  # (in points) small
        self.cluster_width = cluster_width if cluster_width >= 0 else self.auto_cluster_width()  # (in points) from user

        self.curve_offset = self.xseparation * self.line_curviness                               # (in points)

        self.font_size = font_size if font_size >= 0 else self.xseparation * 0.6                 # (in points) from user

        self.height = 0  # (in points) computed automatically from data
        self.width = 0  # (in points) computed automatically from data

        self.default_line_color = (1., 0., 0., 1.)  # (0, 0.4, 0.8, 1) # r, g, b, a
        self.default_cluster_color = (0., 0., 0., 1.)  # r, g, b, a

    def auto_line_width(self) -> float:
        """Calculates automatically the largest possible line width.

         The line width is calculated such that the outgoing or incoming connection can only be as big as the cluster.
        Assumes that the connection width is linear.

        :return: The maximum reasonable line width
        """
        max_line_width = float('inf')
        for cluster in self.clusters:
            line_width_in = line_width_out = float('inf')
            if cluster.insize > 0:
                line_width_in = cluster.draw_size / (cluster.insize + self.line_spacing * (len(cluster.incoming)-1))
            if cluster.outsize > 0:
                line_width_out = cluster.draw_size / (cluster.outsize + self.line_spacing * (len(cluster.outgoing)-1))
            max_line_width = min(max_line_width, line_width_in, line_width_out)
        return max_line_width

    def auto_cluster_width(self) -> float:
        """Simple default cluster width

        :return: width that is 5% of the xseparation
        """
        return self.xseparation * 0.05

    def build_clusters(self, height_method: str) -> Tuple[List[SugiyamaCluster], List[List[SugiyamaCluster]]]:
        """Create Sugiyamaclusters from the underlying graph

        :param height_method: the string to pass to the cluster for it to determine its vertical size
        :return: List of all clusters, first flattened, second in layers
        :rtype: str
        """
        layers = [
            [SugiyamaCluster(self.g.layers[t][i], height_method)
             for i in range(len(self.g.layers[t]))
             ]
            for t in range(self.g.num_steps)
        ]

        for layer in layers:
            for cluster in layer:
                cluster.build()

        clusters = [x for t in range(self.num_layers) for x in layers[t]]

        return clusters, layers

    ####################################################################################################################
    # -------------------------------------------- Helper Functions -------------------------------------------------- #
    ####################################################################################################################

    def pred(self, c: SugiyamaCluster) -> Optional[SugiyamaCluster]:
        """Returns the predecessor of this cluster (e.g. the cluster with rank-1)
        
        :param c: Cluster to find the predecessor of
        :return: SugiyamaCluster or None if no predecessor exists
        """
        if c.rank == 0:
            return None
        return self.ordered[c.tc.layer][c.rank - 1]

    def succ(self, c: SugiyamaCluster) -> Optional[SugiyamaCluster]:
        """Returns the successor of this cluster (e.g. the cluster with rank+1)

        :param c: Cluster to find the successor of
        :return: SugiyamaCluster or None if no successor exists
        """
        if c.rank == len(self.layers[c.tc.layer]) - 1:
            return None
        return self.ordered[c.tc.layer][c.rank + 1]

    @staticmethod
    def num_shared_neighbours(u, v):
        return len(set(v.incoming.keys()).intersection(set(u.incoming.keys()))) + len(
            set(v.outgoing.keys()).intersection(set(u.outgoing.keys())))

    ####################################################################################################################
    # --------------------------------------------- Order Functions -------------------------------------------------- #
    ####################################################################################################################

    def reset_order(self) -> None:
        """Reset the ordering properties of this graph and invalidate successive steps done with previous ordering.

        Clusters are first reset to the position in the original graph and then sorted by supercluster as a base case.
        """
        self.ordered = [self.layers[t].copy() for t in range(self.num_layers)]
        for layer in self.ordered:
            for i, cluster in enumerate(layer):
                cluster.rank = i
        self.sort_by_supercluster()
        self.is_ordered = False
        self.is_aligned = False
        self.is_located = False

    def sort_by_supercluster(self) -> None:
        """Initialize the ranks of clusters to be near other clusters they are connected to

        This works by building superclusters with flood fill. In this case we start new superclusters with the
        highest unplaced cluster, instead of the leftmost one, because this balances much better
        """
        pointers = [0]*self.num_layers
        seen = set()
        max_layer_size = max(map(len, self.layers))

        # traverse all clusters top to bottom instead of left to right
        for i in range(max_layer_size):
            for j in range(self.num_layers):
                if i >= len(self.layers[j]) or i < pointers[j]:
                    continue
                cluster = self.ordered[j][i]

                if cluster in seen:
                    continue
                seen.add(cluster)

                # Start new supercluster and fill it with a flood fill
                supercluster = []

                q = deque()
                q.append(cluster)
                while len(q) > 0:
                    current = q.pop()

                    supercluster.append(current)
                    for nb in current.neighbours:
                        if nb not in seen:
                            q.append(nb)
                            seen.add(nb)

                for scluster in supercluster:
                    l = scluster.tc.layer
                    self.swap_clusters(scluster, self.ordered[l][pointers[l]])
                    pointers[l] += 1

    def set_order(self, barycenter_passes: int = 10) -> None:
        """Order the clusters in self.ordered with the barycenter method

        For each pass of the ordering, the barycenter method is applied once forward and once backward.
        After each pass the ordering is checked whether it changed w.r.t. the previous. All passes are independent
        So we can abort if they are the same.

        :param barycenter_passes: The maximum number of times to repeat the barycenter procedure
        """
        # Make copy to compare if ordering has stabilized
        orders_tmp = [order.copy() for order in self.ordered]

        # Keep doing passes until the maximum number has been reached or the order does no longer change
        for i in range(barycenter_passes):
            print(f"Pass #{i}")
            self._barycenter()
            # self._barycenter()

            if orders_tmp == self.ordered:
                print("Order stabilized")
                break

            orders_tmp = [order.copy() for order in self.ordered]

        self.is_ordered = True

    @staticmethod
    def _bary_rank_layer(layer: List[SugiyamaCluster], max_inrank: int, max_outrank: int, alpha: float = 0.5) -> None:
        """Perform a barycenter sorting on this layer.

        For each cluster the weighted average rank of all incoming and outgoing connections is calculated.
        Then the ordering is decided by a combination of the inrank and outrank, determined by the total size of the
        connections and the alpha factor. If one side has many connections over the other, the side with many
        connections carries more weight. The alpha is the distribution factor, am alpha cose to 1 prioritizes incoming
        connections, an alpha close to 0 prioritizes outgoing. If the alpha is exactly 0 or 1 it is more difficult to
        separate to cluster with the same incoming or outgoing ocnnections.
        If a cluster has no incoming connections it will get the average of the in values of the previous and next
        cluster that do have a valid incoming value. If there is no successive cluster that has a valid value, they are
        given the maximum rank of the previous layer, plus 0.5. Same for the outgoing rank.

        :param layer: the layer to apply the procedure for
        :param max_inrank: The maximum rank in the previous layer
        :param max_outrank:  The maximum rank in the next layer
        :param alpha: Weight factor between 0 and 1 that balances incoming and outgoing
        """
        for cluster in layer:
            cluster.update_cluster_ranks()
            # cluster.update_cluster_ranks_median()

        start_inr = 0
        prev_inr = 0.
        start_outr = 0
        prev_outr = 0.
        for i in range(len(layer)):

            if layer[i].inrank >= 0.:
                for j in range(start_inr, i):
                    layer[j].inrank = (prev_inr + layer[i].inrank) / 2.
                start_inr = i + 1
                prev_inr = layer[i].inrank

            if layer[i].outrank >= 0.:
                for j in range(start_outr, i):
                    layer[j].outrank = (prev_outr + layer[i].outrank) / 2.
                start_outr = i+1
                prev_outr = layer[i].outrank

        for j in range(start_inr, len(layer)):
                layer[j].inrank = max_inrank + 0.5

        for j in range(start_outr, len(layer)):
                layer[j].outrank = max_outrank + 0.5

        total_outsize = sum(map(lambda x: x.outsize, layer))
        total_insize = sum(map(lambda x: x.insize, layer))
        layer.sort(key=lambda c: (c.inrank * total_insize * alpha + c.outrank * total_outsize * (1. - alpha)))

        # layer.sort(key=lambda c: (c.inr * alpha + c.outr * (1.-alpha)))  # simpler
        for i, cluster in enumerate(layer):
            cluster.rank = i

    def _barycenter(self) -> None:
        """Perform barycenter ordering once forward once backward

        Since each layer ordering depends on the previous layer, in the forward iteration layer 0 remains unchanged
        and in the backwards iteration the last layer is unchanged. So for a full ordering both forwards and backwards
        is needed.
        """
        for i in range(self.num_layers):
            layer = self.ordered[i]
            prev_layer_size = (len(self.ordered[i - 1]) - 1) if i > 0 else 0
            next_layer_size = (len(self.ordered[i + 1]) - 1) if i < self.num_layers - 1 else 0

            self._bary_rank_layer(layer, prev_layer_size, next_layer_size, alpha=0.999)

        for i in range(self.num_layers-2, -1, -1):
            layer = self.ordered[i]
            prev_layer_size = (len(self.ordered[i - 1]) - 1) if i > 0 else 0
            next_layer_size = (len(self.ordered[i + 1]) - 1) if i < self.num_layers - 1 else 0

            self._bary_rank_layer(layer, prev_layer_size, next_layer_size, alpha=0.001)

    def swap_clusters(self, cluster1: SugiyamaCluster, cluster2: SugiyamaCluster):
        order = self.ordered[cluster1.tc.layer]
        order[cluster1.rank], order[cluster2.rank] = order[cluster2.rank], order[cluster1.rank]
        cluster1.rank, cluster2.rank = cluster2.rank, cluster1.rank

    ####################################################################################################################
    # ---------------------------------------------- Crossing functions ---------------------------------------------- #
    ####################################################################################################################

    # IMPORTANT: These functions do not consider the size of the connection
    # Which is not really a problem, since one crossing with one larger line is not as bad as
    # many crossings with small lines.
    # The main ordering is done with barycenter, these functions can be used for smaller optimizations

    @staticmethod
    def _compare_ranked_lists(upper: List[int], lower: List[int]):
        """Compare two ordered lists to see how many crossings they have

        Lists are assumed to be sorted low to high and contain ranks. A crossing is when a connection in the
        lower list is above a connection in the higher list.

        :param lower: List of ranks associated with the lower cluster.
        :param upper: List of ranks associated with the higher cluster
        """
        j = i = crossings = 0

        while j < len(lower) and i < len(upper):
            if upper[i] > lower[j]:
                j += 1
            elif upper[i] <= lower[j]:
                crossings += j
                i += 1
        crossings += (len(upper) - i) * j

        return crossings

    @classmethod
    def get_num_crossings(cls, cluster1: SugiyamaCluster, cluster2: SugiyamaCluster):
        """Count the number of crossings these 2 cluster have with each other.

        If cluster1 is upper (lower rank) than cluster2, it will return the current number of crossings.
        If cluster2 is upper, it will return the number of crossings as if they were swapped.

        :param cluster1: cluster that is assumed to be the upper cluster
        :param cluster2: cluster that is assumed to be the lower cluster
        """
        sins_upper = sorted(map(lambda x: x[0].rank, cluster1.incoming.items()))
        sins_lower = sorted(map(lambda x: x[0].rank, cluster2.incoming.items()))
        souts_upper = sorted(map(lambda x: x[0].rank, cluster1.outgoing.items()))
        souts_lower = sorted(map(lambda x: x[0].rank, cluster2.outgoing.items()))
        return (cls._compare_ranked_lists(sins_upper, sins_lower)
                + cls._compare_ranked_lists(souts_upper, souts_lower))

    def crossing_diff_if_swapped(self, cluster1: SugiyamaCluster, cluster2: SugiyamaCluster) -> int:
        """Determine the relative difference in crossings if we were to swap the rank of these clusters.

        The clusters should be of the same layer.

        :param cluster1: One cluster
        :param cluster2: The other cluster
        :return: The relative difference. Negative indicates a decrease in crossings
        """
        if cluster1.rank > cluster2.rank:
            upper, lower = cluster2, cluster1
        else:
            upper, lower = cluster1, cluster2
        return self.get_num_crossings(lower, upper) - self.get_num_crossings(upper, lower)

    ####################################################################################################################
    # ------------------------------------------- Alignment Functions ------------------------------------------------ #
    ####################################################################################################################

    def reset_alignments(self):
        for cluster in self.clusters:
            cluster.reset_alignment()
        self.is_aligned = False
        self.is_located = False

    def set_alignment(self, direction_flag=FORWARD, max_chain=-1, max_inout_diff=2., stairs_iterations=2):

        # Instead of passing on dozens of parameters, this checks if the user has already called the necessary functions
        # if not, it is called with the default parameters
        if not self.is_ordered:
            self.set_order()

        if direction_flag:
            layer_range = range(1, self.num_layers)
        else:
            layer_range = range(self.g.num_steps - 2, -1, -1)

        for layer in layer_range:
            r = -1

            for cluster in self.ordered[layer]:
                if (cluster.insize == 0 or max_inout_diff < 0. or
                        cluster.largest_outgoing / cluster.largest_incoming > max_inout_diff):
                    continue

                # Find cluster in previous layer this one wants to connect to and the weight of the connection
                wanted, connsize = cluster.largest_median_connection(direction=direction_flag)

                if wanted is not None and (max_chain < 0 or wanted.root.chain_length <= max_chain):

                    # Check if this connection contradicts another alignment
                    # priority to the new connection is only given if the weight is higher than all crossings
                    if wanted.rank <= r:
                        if self.has_larger_crossings(wanted, r, connsize):
                            continue
                        self.remove_alignments(wanted, r)

                    wanted.align_with(cluster)
                    r = wanted.rank

        for _ in range(stairs_iterations):
            self.collapse_stairs_iteration()

        self.is_aligned = True

    def has_larger_crossings(self, start_cluster, until_rank, connection_size):
        """Checks for crossing of at least a certain size until it is found or a certain rank is reached

        :param start_cluster: First cluster to check connections from. Should have a lower rank than until_rank
        :param until_rank: Continue up to and including the cluster of this rank.
        :param connection_size: Threshold for which to check.
        """
        cluster = start_cluster
        while cluster is not None and cluster.rank <= until_rank:
            if cluster.align != cluster.root and len(cluster.neighbours[cluster.align]) > connection_size:
                return True
            cluster = self.succ(cluster)
        return False

    def remove_alignments(self, start_cluster: SugiyamaCluster, until_rank: int):
        """Removes all alignments of clusters from start_cluster until a certain rank is reached

        :param start_cluster: first cluster to remove alignment from. Should have a lower rank than until_rank
        :param until_rank: continue up to and including the cluster of this rank.
        """
        cluster = start_cluster
        while cluster is not None and cluster.rank <= until_rank:
            if cluster.align != cluster.root:  # cluster must not be an endpoint
                cluster.align.reset_alignment()
                cluster.reset_endpoint()
            cluster = self.succ(cluster)

    def adjacent_alignments(self, upper: SugiyamaCluster, lower: SugiyamaCluster):
        """Checks whether two aligments are entirely adjacent or that there exists an alignment in between

        :param upper: cluster in the upper (lower rank) alignment
        :param lower: Cluster in the lower (higher rank) alignment
        :return: True if adjacent, False if there exists an alignment in between
        """

        # Set base values
        uroot = upper.root
        lroot = lower.root
        upper = uroot
        lower = lroot

        # Align start layers, at most one while loop will actually run
        while upper.tc.layer < lower.tc.layer:
            upper = upper.align

        while lower.tc.layer < upper.tc.layer:
            lower = lower.align

        # Walk along alignment until either a cluster is found in between or the end of either alignment is reached
        while True:
            if not self.pred(lower) == upper:
                return False
            upper = upper.align
            lower = lower.align
            if upper == uroot or lower == lroot:
                break

        return True

    def crossing_diff_if_swapped_align(self, upper: SugiyamaCluster, lower: SugiyamaCluster):
        """Count the amount of extra crossings this swap would cause. upper and lower should be in adjacent alignments

        Function works by swapping each element and summing the individual differences in crossings

        :param upper: cluster in the upper (lower rank) alignment
        :param lower: Cluster in the lower (higher rank) alignment
        """
        uroot = upper.root
        lroot = lower.root
        cluster = lroot
        crossing_diff = 0
        length = 0
        while True:
            predecessor = self.pred(cluster)
            if predecessor is not None and predecessor.root == uroot:
                crossing_diff += self.crossing_diff_if_swapped(cluster, predecessor)
                length += 1
            cluster = cluster.align
            if cluster == lroot:
                break
        return crossing_diff - 2*(length-1)

    def swap_align(self, upper: SugiyamaCluster, lower: SugiyamaCluster):
        """Count the amount of extra crossings this swap would cause. upper and lower should be in adjacent alignments

        Function works by swapping each element and summing the individual differences in crossings

        :param upper: cluster in the upper (lower rank) alignment
        :param lower: Cluster in the lower (higher rank) alignment
        """
        uroot = upper.root
        lroot = lower.root
        cluster = lroot
        while True:
            predecessor = self.pred(cluster)
            if predecessor is not None and predecessor.root == uroot:
                self.swap_clusters(cluster, predecessor)

            cluster = cluster.align
            if cluster == lroot:
                break

    def collapse_stairs_iteration(self, minimum_want=3, allowed_extra_crossings=0):
        """Mitigates the staircase effect on connected parts of the graph

        The goal is to move shorter chains closer to their desired position. Shorter chains have a better chance to
        fit tightly as opposed to longer chains.
        For every two adjacent alignments, it is always that either root is a predecessor or successor to a cluster
        in the other alignment. As such we only need to check the roots, but we have to check both the direction this
        alignment wants to go and the direction the predecessor alignment and successor alignment of the root want to go
        We first establish the desired direction of each alignment in its root. Then for each root we check 4 cases:
        case 1: this alignment wants to go up
        case 2: this alignment wants to go down
        case 3: the successor wants to go up
        case 4: the predecessor wants to go down
        These cases are mutually exclusive, even though case 1 and 4 cause the same swap, as do 2 and 3.
        Preference is given to case 1 and 2. It is a possibility that both case 3 and 4 are valid, but after execution
        of either case, the other will be obstructed, because the predecessor and successor change. Cases are looped
        until a stop condition is reached. In short for a swap to occur the following things must hold:
        a. in case 1 and 2, this chain must be shorter than the predecessor/successor. For case 3 and 4 this is inverted
        b. the shorter chain must move in its desired direction, updated after every step.
            A direction value between 2 and -2 (inclusive) is optimal and is treated as no desired direction.
        c. the alignments must be fully adjacent (ask me if this requirement is unclear)
        d. the swap may not increase the amount of crossings in the graph more than a certain amount (default: 0)
        """
        for cluster in self.clusters:
            if cluster.root != cluster:
                continue
            cluster.update_wanted_direction()

        for cluster in self.clusters:
            if cluster.root != cluster:
                continue

            successor = self.succ(cluster)
            predecessor = self.pred(cluster)

            # Case 1
            while (cluster.wanted_direction <= -minimum_want
                   and predecessor is not None and predecessor.root.chain_length > cluster.chain_length
                   and self.adjacent_alignments(predecessor, cluster)
                   and self.crossing_diff_if_swapped_align(predecessor, cluster) <= allowed_extra_crossings):
                self.swap_align(predecessor, cluster)
                predecessor.root.update_wanted_direction()
                cluster.update_wanted_direction()
                predecessor = self.pred(cluster)

            # Case 2
            while (cluster.wanted_direction >= minimum_want
                    and successor is not None and successor.root.chain_length > cluster.chain_length
                    and self.adjacent_alignments(cluster, successor)
                    and self.crossing_diff_if_swapped_align(cluster, successor) <= allowed_extra_crossings):

                self.swap_align(cluster, successor)
                successor.root.update_wanted_direction()
                cluster.update_wanted_direction()
                successor = self.succ(cluster)

            # Case 3
            while (successor is not None and successor.root.chain_length <= cluster.chain_length
                   and successor.root.wanted_direction < -minimum_want
                   and self.adjacent_alignments(cluster, successor)
                   and self.crossing_diff_if_swapped_align(cluster, successor) <= allowed_extra_crossings):

                self.swap_align(cluster, successor)
                successor.root.update_wanted_direction()
                cluster.update_wanted_direction()
                successor = self.succ(cluster)

            # Case 4
            while (predecessor is not None and predecessor.root.chain_length < cluster.chain_length
                    and predecessor.root.wanted_direction >= minimum_want
                    and self.adjacent_alignments(predecessor, cluster)
                    and self.crossing_diff_if_swapped_align(predecessor, cluster) <= allowed_extra_crossings):

                self.swap_align(predecessor, cluster)
                predecessor.root.update_wanted_direction()
                cluster.update_wanted_direction()
                predecessor = self.pred(cluster)

    ####################################################################################################################
    # ------------------------------------------- Location Functions ------------------------------------------------- #
    ####################################################################################################################

    def set_locations(self, averaging_iterations=5):

        # Instead of passing on dozens of parameters, this checks if the user has already called the necessary functions
        # if not, it is called with the default parameters
        if not self.is_aligned:
            self.set_alignment()

        self.set_x_positions()

        for cluster in self.clusters:
            if cluster.root == cluster:
                self.place_block(cluster)

        self.set_y_positions()

        for _ in range(averaging_iterations):
            for cluster in self.clusters:
                cluster._y = -1.

            for cluster in self.clusters:
                if cluster.root == cluster:
                    self.avg_block(cluster)

            self.set_y_positions()

        self.check_locations()
        self.is_located = True

    def check_locations(self, excepting=True):
        for order in self.ordered:
            prev = -1
            for cluster in order:
                if cluster.y <= prev:
                    if excepting:
                        raise UnorderedException(f"Locations are not in order of ranks: {cluster} is higher than {self.pred(cluster)}")
                    else:
                        print(f"{cluster} is higher than {self.pred(cluster)}")
                prev = cluster.y

    def center_distance(self, u: SugiyamaCluster, v: SugiyamaCluster, non_connectedness_factor=1.):
        """Calculate the distance between the centers of two cluster if they were to be adjacent

        :param u: Cluster
        :param v: Cluster
        :param non_connectedness_factor: Factor by which to increase the distance if the clusters do not share neighbors
        """
        if non_connectedness_factor != 1. and self.num_shared_neighbours(u, v) == 0:
            return self.yseparation * non_connectedness_factor + (u.draw_size + v.draw_size) / 2.
        return self.yseparation + (u.draw_size + v.draw_size) / 2.

    def place_block(self, root):
        """Place an aligned section by placing all blocks above it and then fitting it as high as possible

        :param root: Root of the alignment to place
        """

        # If block was already placed, skip
        if root._y < 0.:
            root._y = 0.
            cluster = root

            while True:
                if cluster.rank > 0:
                    predecessor = self.pred(cluster)
                    self.place_block(predecessor.root)

                    root._y = max(root._y, predecessor.root._y + self.center_distance(predecessor, cluster))

                cluster = cluster.align
                if cluster == root:
                    break

    def avg_block(self, root):
        """Place an aligned section by taking the average position

        place_block() puts all sections as high as they will go from top to bottom. This creates an upper bound on the
        alignment. Now from bottom to top we take the position this alignment would be placed based on average,
        bounded by the upper bound already calculated and the lower bound by the average of the block below.
        """
        if root._y < 0.:
            root._y = 0.
            cluster = root
            upper_bound = root.y
            lower_bound = float('inf')
            total = 0.
            ctr = 0

            while True:
                successor = self.succ(cluster)
                if successor is not None:
                    self.avg_block(successor.root)

                    lower_bound = min(lower_bound, successor.root._y - self.center_distance(cluster, successor))

                for k, value in cluster.outgoing.items():
                    if k == cluster.align:
                        continue
                    ctr += len(value)
                    total += k.y * len(value)

                for k, value in cluster.align.incoming.items():
                    if k == cluster:
                        continue
                    ctr += len(value)
                    total += k.y * len(value)

                cluster = cluster.align
                if cluster == root:
                    break
            if ctr == 0:
                root._y = upper_bound
            else:
                root._y = max(upper_bound, min(lower_bound, total / ctr))

    def set_x_positions(self):
        for cluster in self.clusters:
            cluster.x = self.xmargin + self.xseparation * cluster.tc.layer
        self.width = 2*self.xmargin + self.xseparation * self.num_layers

    def set_y_positions(self):
        min_y = min(map(lambda x: x.root._y - x.draw_size / 2., self.clusters))
        for cluster in self.clusters:
            cluster.y = cluster.root._y + self.ymargin - min_y

        self.height = max(map(lambda x: x.root.y + x.draw_size / 2., self.clusters)) + self.ymargin

    ####################################################################################################################
    # ------------------------------------------ Statistics Functions ------------------------------------------------ #
    ####################################################################################################################

    def streak_below(self, data, num):
        longest = 0
        current = 0
        for d in data:
            if d < num:
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 0
        return longest

    def streak_no_cross(self, data, num):
        longest = 1
        current = 1
        crossings = 0
        for i in range(len(data)-1):
            if (data[i] < num) == (data[i+1] < num):
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 1
                crossings += 1
        return longest, crossings

    def stat_surface(self, data):
        h = self.height / 3
        surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, (0, 0, self.width, h*len(data)))
        context = cairo.Context(surface)

        marg = 0.1

        for i, name in enumerate(data):

            if name == "in_out_difference":
                d = self.g.layer_in_out_diff()
            elif name == "layer_num_clusters":
                d = self.g.layer_num_clusters()
            elif name == "layer_num_members":
                d = self.g.layer_num_members()
            elif name == "homogeneity":
                d = self.g.homogeneity()
            elif name == "homogeneity_diff":
                d = self.g.homogeneity_diff()
            else:
                return

            maxval = max(d)
            scale = (h * (1. - 2*marg)) / maxval

            context.set_source_rgba(0, 0, 0, 1)
            context.move_to(self.xmargin*0.95, (i + marg)*h)
            context.line_to(self.xmargin*0.9,  (i + marg)*h)
            context.line_to(self.xmargin*0.9, (i + 1. - marg)*h)
            context.line_to(self.width - self.xmargin, (i + 1. - marg)*h)
            context.stroke()

            text_to_show = f"{maxval:.2f}"
            context.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            context.set_font_size(self.font_size*0.4)
            _, _, tw, th, _, _ = context.text_extents(text_to_show)
            context.move_to(self.xmargin*0.85 - tw, (i + marg)*h + th)
            context.show_text(text_to_show)

            avg = sum(d)/len(d)
            context.set_source_rgba(0, 1, 0, 1)
            context.move_to(self.xmargin, (i+marg)*h + (maxval - avg)*scale)
            context.line_to(self.width - self.xmargin, (i + marg) * h + (maxval - avg) * scale)
            context.stroke()

            context.set_source_rgba(0, 0, 1, 1)

            context.move_to(self.xmargin, (i + marg)*h + (maxval - d[0])*scale)
            for j in range(1, len(d)):
                context.line_to(self.xmargin + j*self.xseparation, (i + marg)*h + (maxval - d[j])*scale)

            context.stroke()

            streak, crossings = self.streak_no_cross(d, avg)
            text_to_show = f"{name}: streak ({streak}), crossings ({crossings}), cross_percent ({(crossings/len(d)):2f})"

            context.set_font_size(self.font_size*0.5)
            _, _, tw, th, _, _ = context.text_extents(text_to_show)
            context.move_to(0.3 * self.xseparation + self.xmargin, (i + marg)*h + th)
            context.show_text(text_to_show)

        surface.flush()
        return surface

    ####################################################################################################################
    # ------------------------------------------- Drawing Functions -------------------------------------------------- #
    ####################################################################################################################

    def calculate_line_origins(self, source_cluster):
        """Calculate the absolute origins for each line incident to this cluster

        This is calculated by sorting the other endpoints and "empirically" set them
        one after the other around the center.
        """
        incomings = list(source_cluster.incoming.items())
        incomings.sort(key=lambda x: x[0].rank)
        outgoings = list(source_cluster.outgoing.items())
        outgoings.sort(key=lambda x: x[0].rank)

        # The key is a pair of (source, target)
        # The value is the y coordinate of where the endpoint in source should originate.
        pairs = {}

        # Space to leave between each line
        line_separation = self.line_width * self.line_spacing

        # We calculate around the center of the cluster, so this is the offset to start at
        half_in_width = (source_cluster.insize * self.line_width + line_separation * (len(incomings) - 1)) / 2.

        # source_cluster.y refers to the center of the cluster, so subtract half the width such that the center align
        cumulative = source_cluster.y - half_in_width
        for (target_cluster, members) in incomings:
            thickness = len(members) * self.line_width
            pairs[(source_cluster, target_cluster)] = cumulative + thickness / 2.
            cumulative += thickness + line_separation

        # We calculate around the center of the cluster, so this is the offset to start at
        half_out_width = (source_cluster.outsize * self.line_width + line_separation * (len(outgoings) - 1)) / 2.

        # source_cluster.y refers to the center of the cluster, so subtract half the width such that the center align
        cumulative = source_cluster.y - half_out_width
        for (target_cluster, members) in outgoings:
            thickness = len(members) * self.line_width
            pairs[(source_cluster, target_cluster)] = cumulative + thickness / 2.
            cumulative += thickness + line_separation

        return pairs

    def draw_line(self, source: SugiyamaCluster, target: SugiyamaCluster,
                  line_coordinates: dict, colormap: dict,
                  context: cairo.Context,
                  show_annotations=False):
        """Draws the connection between source and target with label colors

        The line is divided in sections according to the labels of the nodes in this connection.
        For each unique label a line is drawn in the associated color with exact offset as to seem one single cohesive

        :param source: One endpoint of the line
        :param target: The other endpoint of the line
        :param line_coordinates: Dictionary with absolute coordinates of the endpoints
        :param colormap: Dictionary with label names and associated colors
        :param context: cairo context to draw this line on
        :param show_annotations: provides annotations when a line changes size significantly
        """
        members = source.neighbours[target]
        num_members = len(members)
        thickness = num_members*self.line_width
        half_thickness = len(members) * self.line_width / 2.

        labels = Counter(map(lambda x: x.name, members))
        labels = [(colormap.get(lbl, self.default_line_color), cnt/num_members) for (lbl, cnt) in sorted(list(labels.items()))]

        y_source = line_coordinates[(source, target)]
        y_target = line_coordinates[(target, source)]
        context.save()
        context.rectangle(source.x, 0, target.x, self.height)
        context.clip()

        if len(labels) == 1:
            (r, g, b, a) = labels[0][0]
            context.set_source_rgba(r, g, b, a)
            context.set_line_width(thickness)
            context.move_to(source.x, y_source)
            context.curve_to(source.x + self.curve_offset, y_source,
                             target.x - self.curve_offset, y_target,
                             target.x, y_target)

            context.stroke()

        else:
            coloured_bezier(context,
                            (source.x, y_source),
                            (source.x + self.curve_offset, y_source),
                            (target.x - self.curve_offset, y_target),
                            (target.x, y_target),
                            labels,
                            thickness,
                            detail=min(100, max(4, int(abs(y_target-y_source)*self.line_width))))

        if show_annotations:
            # draw some annotatations of the bundle size
            context.set_source_rgb(0, 0, 0)
            context.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            context.set_font_size(min(self.line_width * len(members), self.xseparation*0.7))  # in user space units

            if len(members) >= 1.5 * max(map(len, source.incoming.values()), default=0.):
                to_print = str(len(members))
                _, _, width, height, _, _ = context.text_extents(to_print)
                context.move_to(source.x + self.xseparation / 2 - width / 2,
                                (y_source + y_target) / 2 + height / 2)
                context.show_text(to_print)
                context.stroke()
        context.restore()

    def fade_cluster(self, ctx: cairo.Context, cluster: SugiyamaCluster, colormap: Dict, direction):
        colors = [(colormap.get(lbl, self.default_line_color), cnt/len(cluster)) for (lbl, cnt) in sorted(list(Counter(map(lambda x: x.name, cluster.members)).items()))]
        if direction:
            coloured_bezier(ctx,
                            (cluster.x - self.xseparation / 3., cluster.y),
                            (cluster.x, cluster.y),
                            (cluster.x, cluster.y),
                            (cluster.x, cluster.y),
                            colors=colors, width=self.line_width*len(cluster), detail=4, fade='in')
        else:
            coloured_bezier(ctx,
                            (cluster.x, cluster.y),
                            (cluster.x, cluster.y),
                            (cluster.x, cluster.y),
                            (cluster.x + self.xseparation / 3., cluster.y),
                            colors=colors, width=self.line_width * len(cluster), detail=4, fade='out')

    def timestamp_surface(self, timestamp_translator):
        surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        context = cairo.Context(surface)
        context.set_source_rgb(0, 0, 0)
        context.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(self.font_size)  # in user space units # self.xseparation * 0.6

        for t in range(self.num_layers):
            context.move_to(self.xmargin + t * self.xseparation, 0)
            context.line_to(self.xmargin + t * self.xseparation, self.xseparation * 0.2)

            to_print = str(t)
            if timestamp_translator:
                to_print = timestamp_translator[t]

            _, _, width, height, _, _ = context.text_extents(to_print)

            context.move_to(self.xmargin + t * self.xseparation + height / 2,
                            width + self.xseparation * 0.3)

            context.save()
            context.rotate(math.radians(270))  # angle in rad
            context.show_text(to_print)
            context.restore()

        context.stroke()
        surface.flush()
        print(surface.ink_extents())
        return surface

    def connection_surface(self, colormap, show_annotations, fading):
        surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        context = cairo.Context(surface)
        line_coordinates = dict()  # k: (source, target), v: y-coordinate of endpoint in source.

        for cluster in self.clusters:
            line_coordinates.update(self.calculate_line_origins(cluster))

        if fading:
            for cluster in self.clusters:
                if cluster.insize == 0:
                    self.fade_cluster(context, cluster, colormap, direction=INCOMING)
                if cluster.outsize == 0:
                    self.fade_cluster(context, cluster, colormap, direction=OUTGOING)

        already_drawn = set()

        for (source, target) in line_coordinates.keys():
            if (target, source) in already_drawn:
                continue

            self.draw_line(source, target, line_coordinates, colormap, context, show_annotations=show_annotations)
            # self.draw_line_monochrome(source, target, line_coordinates, context)

            already_drawn.add((source, target))
            already_drawn.add((target, source))

        surface.flush()
        print(surface.ink_extents())
        return surface

    def cluster_surface(self):
        surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        context = cairo.Context(surface)
        (r, g, b, a) = self.default_cluster_color

        context.set_line_width(self.cluster_width)
        context.set_source_rgba(r, g, b, a)

        for cluster in self.clusters:
            cx, cy = cluster.pos()
            context.move_to(cx, cy - cluster.draw_size / 2.)
            context.line_to(cx, cy + cluster.draw_size / 2.)
            context.stroke()
        surface.flush()
        print(surface.ink_extents())
        return surface

    def debug_surface(self, debug_info: Set[str]):
        surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        context = cairo.Context(surface)

        context.set_source_rgb(0.2, 0.2, 0.2)
        context.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(self.font_size * 0.2)

        for cluster in self.clusters:
            cx, cy = cluster.pos()
            if cluster.root == cluster:
                if "swap_above" in debug_info:
                    context.move_to(cx + self.xseparation * 0.1, cy)
                    context.show_text(
                        f"{self.crossing_diff_if_swapped_align(self.pred(cluster), cluster) if self.pred(cluster) is not None else 0:.2f}")

            if "ranks" in debug_info:
                context.move_to(cx + self.xseparation * 0.1, cy)
                context.show_text(f"{cluster.outrank:.2f}")
                context.move_to(cx - self.xseparation * 0.3, cy)
                context.show_text(f"{cluster.inrank:.2f}")

            if "id" in debug_info:
                context.move_to(cx + self.xseparation * 0.1, cy)
                context.show_text(f"{cluster.tc.id}")
                context.move_to(cx - self.xseparation * 0.3, cy)
                context.show_text(f"{cluster.tc.layer}")
        surface.flush()
        return surface

    def paint_surface(self, context, to_draw, x=0, y=0):
        context.save()
        context.set_source_surface(to_draw, x, y)
        context.paint()
        context.restore()

    def draw_graph(self, filename: str = "output/example.svg",
                   colormap=None,
                   show_timestamps=True, timestamp_translator=None,
                   show_annotations=False, debug_info=None, stats_info=None,
                   fading=False):

        if colormap is None:
            colormap = dict()

        # Instead of passing on dozens of parameters, this checks if the user has already called the necessary functions
        # if not, it is called with the default parameters
        if not self.is_located:
            self.set_locations()

        surfaces = dict()

        surfaces["conn"] = self.connection_surface(colormap, show_annotations=show_annotations, fading=fading)

        surfaces["clus"] = self.cluster_surface()

        if debug_info is not None:
            surfaces["debu"] = self.debug_surface(debug_info=debug_info)

        offset = 0

        if show_timestamps:
            timesurf = self.timestamp_surface(timestamp_translator)
            _, _, _, timeheight = timesurf.ink_extents()
            offset += self.ymargin + timeheight
            surfaces["time"] = timesurf

        if stats_info is not None:
            statsurf = self.stat_surface(stats_info)
            _, _, _, statheight = statsurf.ink_extents()
            offset += self.ymargin + statheight
            surfaces["stat"] = statsurf

        surface = cairo.SVGSurface(filename,
                                       (self.width + 2*self.xmargin)*self.scale,
                                       (self.height + offset + 2*self.ymargin)*self.scale)
        context = cairo.Context(surface)
        # context.scale(self.scale, self.scale)


        offset = self.height
        self.paint_surface(context, surfaces["conn"])
        self.paint_surface(context, surfaces["clus"])
        if debug_info is not None:
            self.paint_surface(context, surfaces["debu"])
        if show_timestamps:
            y = offset + self.ymargin
            _, _, _, h = surfaces["time"].ink_extents()
            offset += self.ymargin + h
            self.paint_surface(context, surfaces["time"], y=y)
        if stats_info is not None:
            y = offset+self.ymargin
            _, _, _, h = surfaces["stat"].ink_extents()
            offset += self.ymargin + h
            self.paint_surface(context, surfaces["stat"], y=y)
        surface.flush()
        surface.finish()
        for s in surfaces.values():
            s.finish()
        print(offset)

