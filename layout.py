from __future__ import annotations

from collections import Counter
from typing import List

from TimeGraph import TimeGraph, TimeCluster
import cairocffi as cairo
import math

INCOMING = FORWARD = True
OUTGOING = BACKWARD = False


class NotRootException(Exception):
    pass


class UnorderedException(Exception):
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

        self.members = self.tc.members  # Set of TimeNodes in this cluster

        # Order properties
        self.rank = -1
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

    def draw_height(self, method):
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

    def build(self, minimum_cluster_size, minimum_connection_size):
        for c, connection_nodes in self.tc.incoming.items():

            if len(connection_nodes) < minimum_connection_size or len(c) < minimum_cluster_size:
                continue

            self.incoming[c.sc] = connection_nodes
            self.insize += len(connection_nodes)

        for c, connection_nodes in self.tc.outgoing.items():

            if len(connection_nodes) < minimum_connection_size or len(c) < minimum_cluster_size:
                continue

            self.outgoing[c.sc] = connection_nodes
            self.outsize += len(connection_nodes)

        self.neighbours = {**self.incoming, **self.outgoing}

    def get_cluster_ranks(self):
        if self.insize > 0:
            inrank = sum([nb.rank * len(conn) for nb, conn in self.incoming.items()]) / self.insize
        else:
            inrank = self.rank

        if self.outsize > 0:
            outrank = sum([n.rank * len(l) for n, l in self.outgoing.items()]) / self.outsize
        else:
            outrank = self.rank

        return inrank, outrank

    def reset_alignment(self):
        self.root = self
        self.align = self
        self.chain_length = 1

    def reset_endpoint(self):
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
        while ptr < len(connections) and connections[ptr][1] == connections[0][1]:
            ptr += 1

        brother = connections[(ptr - (1 if lower else 0)) // 2][0]
        connsize = len(connections[0][1])
        return brother, connsize

    def align_with(self, next_cluster: SugiyamaCluster):
        self.align = next_cluster
        next_cluster.root = self.root
        next_cluster.align = self.root
        self.root.chain_length += 1

    def update_wanted_direction(self):
        """Function to calculate which direction an alignment would like to move in depending on the ranks of its connections

        :param root: The cluster that is at the root of the alignment.
        :return: The direction (positive is downwards, negative is upwards) and strength of the direction
        """
        if self.root != self:
            raise NotRootException("You can only call wanted_direction on an alignment root.")

        # l holds the alignment plus a buffer of 1 at both sides
        l = []

        # Add the cluster that the root would have aligned with if it was allowed
        l.append(self.largest_median_connection(direction=INCOMING)[0])
        cluster = self
        while True:
            l.append(cluster)
            if cluster.align == self:
                break
            cluster = cluster.align
        # Similarly, add the cluster that the last one would like to align with
        l.append(self.largest_median_connection(direction=OUTGOING)[0])

        total = 0
        for i in range(1, len(l) - 1):
            cluster = l[i]
            # Compare incoming connections to the rank of the previous in the alignment
            for k, v in cluster.incoming.items():
                total += len(v) * (k.rank - l[i - 1].rank)

            # Compare outgoing connections to the rank of the next in the alignment
            for k, v in cluster.outgoing.items():
                total += len(v) * (k.rank - l[i + 1].rank)

        # The first and last elements of l were not part of the alignment for a reason
        # So we have to factor in the direction of where this alignment would have liked to go
        if l[0] is not None and l[0].align.tc.layer == l[1].tc.layer:
            total += l[0].align.rank - l[1].rank

        if l[-1] is not None and l[-1].root.tc.layer <= l[-2].tc.layer:
            prev = l[-1].root
            while prev.align != l[-1]:
                prev = prev.align
            total += prev.rank - l[-2].rank

        self.wanted_direction = total
        return total

    def pos(self):
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
                 minimum_cluster_size=0, minimum_connection_size=0,
                 line_width=-1., line_spacing=0.0,
                 line_curviness=0.3,
                 horizontal_density=1., vertical_density=10,
                 cluster_width=-1,
                 cluster_height_method='sqrt',
                 font_size=-1):

        # Set basic information from the time graph
        self.g = g
        self.num_layers = self.g.num_steps

        # Set different collection objects
        self.clusters, self.layers = self.build_and_trim_clusters(minimum_cluster_size,
                                                                  minimum_connection_size,
                                                                  cluster_height_method)
        self.ordered = []  # Filled by reset_order()
        self.reset_order()

        # Set flags
        self.is_ordered = False
        self.is_aligned = False
        self.is_located = False

        # General info
        max_cluster = max(self.clusters, key=lambda x: x.draw_size)
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

        self.bottom = 0  # (in points) computed automatically from data

        self.default_line_color = (1., 0., 0., 1.)  # (0, 0.4, 0.8, 1) # r, g, b, a
        self.default_cluster_color = (0., 0., 0., 1.)  # r, g, b, a

    def auto_line_width(self):
        max_line_width = float('inf')
        for cluster in self.clusters:
            line_width_in = line_width_out = float('inf')
            if cluster.insize > 0:
                line_width_in = cluster.draw_size / (cluster.insize + self.line_spacing * len(cluster.incoming))
            if cluster.outsize > 0:
                line_width_out = cluster.draw_size / (cluster.outsize + self.line_spacing * len(cluster.outgoing))
            max_line_width = min(max_line_width, line_width_in, line_width_out)
        return max_line_width

    def auto_cluster_width(self):
        return self.xseparation * 0.05

    def build_and_trim_clusters(self, minimum_cluster_size, minimum_connection_size, height_method):

        layers = [
            [SugiyamaCluster(self.g.clusters[t][i], height_method)
             for i in range(len(self.g.clusters[t]))
             if len(self.g.clusters[t][i]) >= minimum_cluster_size
             ]
            for t in range(self.g.num_steps)
        ]

        for layer in layers:
            for cluster in layer:
                cluster.build(minimum_cluster_size, minimum_connection_size)

        layers = [
            [cluster for cluster in layer if cluster.insize + cluster.outsize > 0]
            for layer in layers
        ]

        clusters = [x for t in range(self.num_layers) for x in layers[t]]

        return clusters, layers

    ####################################################################################################################
    # -------------------------------------------- Helper Functions -------------------------------------------------- #
    ####################################################################################################################

    def pred(self, c: SugiyamaCluster) -> SugiyamaCluster:
        """Returns the predecessor of this cluster (e.g. the cluster with rank-1)
        
        :param c: Cluster to find the predecessor of
        :return: SugiyamaCluster or None if no predecessor exists
        """
        if c.rank == 0:
            return None
        return self.ordered[c.tc.layer][c.rank - 1]

    def succ(self, c: SugiyamaCluster) -> SugiyamaCluster:
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

    def reset_order(self):
        self.ordered = [self.layers[t].copy() for t in range(self.num_layers)]
        for layer in self.ordered:
            for i, cluster in enumerate(layer):
                cluster.rank = i
        self.is_ordered = False
        self.is_aligned = False
        self.is_located = False

    def set_order(self, barycenter_passes: int = 10, repetitions_per_pass: int = 5):

        # Make copy to compare if ordering has stabilized
        orders_tmp = [order.copy() for order in self.ordered]

        # Keep doing passes until the maximum number has been reached or the order does no longer change
        for i in range(barycenter_passes):
            print(f"Pass #{i}")
            self._barycenter(FORWARD, repetitions_per_pass)
            self._barycenter(BACKWARD, repetitions_per_pass)

            if orders_tmp == self.ordered:
                print("Order stabilized")
                break

            orders_tmp = [order.copy() for order in self.ordered]

        self.is_ordered = True

    @staticmethod
    def _bary_rank_layer(layer: List[SugiyamaCluster], direction_flag):
        for cluster in layer:
            inr, outr = cluster.get_cluster_ranks()

            if direction_flag:
                cluster.rank = inr
                cluster.secondrank = outr
            else:
                cluster.rank = outr
                cluster.secondrank = inr

        layer.sort(key=lambda c: (c.rank, c.secondrank))
        for i, cluster in enumerate(layer):
            cluster.rank = i

    def _barycenter(self, direction_flag, reps):
        # True means forward pass (leaves first layer unchanged)
        # False is backwards pass (last layer unchanged)
        if direction_flag:
            r = range(1, self.num_layers)
        else:
            r = range(self.num_layers - 2, -1, -1)

        for _ in range(reps):
            for t in r:
                layer = self.ordered[t]
                self._bary_rank_layer(layer, direction_flag)

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
    def _compare_ranked_lists(lower: List[int], upper: List[int]):
        """Compare two ordered lists to see how many crossings they have

        Lists are assumed to be sorted low to high and contain ranks. A crossing is when a connection in the
        lower list is above a connection in the higher list.

        :param lower: List of ranks associated with the lower cluster.
        :param upper: List of ranks associated with the higher cluster
        """
        i = j = crossings = 0

        while i < len(lower) and j < len(upper):
            if upper[j] > lower[i]:
                crossings += len(lower) - i
                i += 1
            elif upper[j] < lower[i]:
                j += 1
            else:
                i += 1
                j += 1

        return crossings

    @classmethod
    def get_num_crossings(cls, cluster1: SugiyamaCluster, cluster2: SugiyamaCluster):
        """Count the number of crossings these 2 cluster have with each other.

        If cluster1 is lower (higher rank) than cluster2, it will return the current number of crossings.
        If cluster2 is lower, it will return the number of crossings as if they were swapped.

        :param cluster1: cluster that is assumed to be the lower cluster
        :param cluster2: cluster that is assumed to be the higher cluster
        """
        sins_lower = sorted(map(lambda x: x[0].rank, cluster1.incoming.items()))
        sins_upper = sorted(map(lambda x: x[0].rank, cluster2.incoming.items()))
        souts_lower = sorted(map(lambda x: x[0].rank, cluster1.outgoing.items()))
        souts_upper = sorted(map(lambda x: x[0].rank, cluster2.outgoing.items()))
        return (cls._compare_ranked_lists(sins_lower, sins_upper)
                + cls._compare_ranked_lists(souts_lower, souts_upper))

    def crossing_diff_if_swapped(self, cluster1: SugiyamaCluster, cluster2: SugiyamaCluster):
        if cluster1.rank > cluster2.rank:
            cluster1, cluster2 = cluster2, cluster1
        return self.get_num_crossings(cluster2, cluster1) - self.get_num_crossings(cluster1, cluster2)

    ####################################################################################################################
    # ------------------------------------------- Alignment Functions ------------------------------------------------ #
    ####################################################################################################################

    def reset_alignments(self):
        for cluster in self.clusters:
            cluster.reset_alignment()
        self.is_aligned = False
        self.is_located = False

    def align_clusters(self, direction_flag=FORWARD, max_chain=-1, stairs_iterations=5):

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
        while True:
            predecessor = self.pred(cluster)
            if predecessor is not None and predecessor.root == uroot:
                crossing_diff += self.crossing_diff_if_swapped(cluster, predecessor)

            cluster = cluster.align
            if cluster == lroot:
                break
        return crossing_diff

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

    def collapse_stairs_iteration(self, minimum_want=2, allowed_extra_crossings=0):
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
        c. the alignments must be fully adjacent (ask me if this requirement is unclear
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
            while (cluster.wanted_direction < -minimum_want
                   and predecessor is not None and predecessor.root.chain_length > cluster.chain_length
                   and self.adjacent_alignments(predecessor, cluster)
                   and self.crossing_diff_if_swapped_align(predecessor, cluster) <= allowed_extra_crossings):
                self.swap_align(predecessor, cluster)
                predecessor.root.update_wanted_direction()
                cluster.update_wanted_direction()
                predecessor = self.pred(cluster)

            # Case 2
            while (cluster.wanted_direction > minimum_want
                    and successor is not None and successor.root.chain_length > cluster.chain_length
                    and self.adjacent_alignments(cluster, successor)
                    and self.crossing_diff_if_swapped_align(cluster, successor) <= allowed_extra_crossings):

                self.swap_align(cluster, successor)
                successor.root.update_wanted_direction()
                cluster.update_wanted_direction()
                successor = self.succ(cluster)

            # Case 3
            while (successor is not None and successor.root.chain_length < cluster.chain_length
                   and successor.root.wanted_direction < -minimum_want
                   and self.adjacent_alignments(cluster, successor)
                   and self.crossing_diff_if_swapped_align(cluster, successor) <= allowed_extra_crossings):

                self.swap_align(cluster, successor)
                successor.root.update_wanted_direction()
                cluster.update_wanted_direction()
                successor = self.succ(cluster)

            # Case 4
            while (predecessor is not None and predecessor.root.chain_length < cluster.chain_length
                    and predecessor.root.wanted_direction > minimum_want
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
            self.align_clusters()

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

    def set_y_positions(self):
        min_y = min(self.clusters, key=lambda x: x.y).y
        for cluster in self.clusters:
            cluster.y = cluster.root._y + cluster.draw_size / 2. + self.ymargin - min_y

        self.bottom = max(map(lambda x: x.y + x.draw_size / 2., self.clusters)) + self.ymargin

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

    def draw_line_monochrome(self, source: SugiyamaCluster, target: SugiyamaCluster, line_coordinates: dict,
                             context: cairo.Context):
        # Same as draw line, but disregards color information
        context.set_source_rgba(1., 0., 0., 1.)
        context.set_line_width(len(source.neighbours[target]) * self.line_width)
        y_source = line_coordinates[(source, target)]
        y_target = line_coordinates[(target, source)]

        context.move_to(source.x, y_source)
        context.curve_to(source.x + self.curve_offset, y_source,
                         target.x - self.curve_offset, y_target,
                         target.x, y_target)

        context.stroke()

    def draw_line(self, source: SugiyamaCluster, target: SugiyamaCluster,
                  line_coordinates: dict, colormap: dict,
                  context: cairo.Context):
        """Draws the connection between source and target with label colors

        The line is divided in sections according to the labels of the nodes in this connection.
        For each unique label a line is drawn in the associated color with exact offset as to seem one single cohesive

        :param source: One endpoint of the line
        :param target: The other endpoint of the line
        :param line_coordinates: Dictionary with absolute coordinates of the endpoints
        :param colormap: Dictionary with label names and associated colors
        :param context: cairo context to draw this line on
        """
        members = source.neighbours[target]
        half_thickness = len(members) * self.line_width / 2.

        labels = Counter(map(lambda x: x.name, members))
        labels = sorted(list(labels.items()))

        cumulative = -half_thickness

        for (label, count) in labels:
            if label in colormap.keys():
                (r, g, b, a) = colormap[label]
            else:
                (r, g, b, a) = self.default_line_color
            context.set_source_rgba(r, g, b, a)
            context.set_line_width(count * self.line_width)

            # Find the center of the line
            offset = cumulative + count * self.line_width / 2.

            y_source = line_coordinates[(source, target)] + offset
            y_target = line_coordinates[(target, source)] + offset

            context.move_to(source.x, y_source)
            context.curve_to(source.x + self.curve_offset, y_source,
                             target.x - self.curve_offset, y_target,
                             target.x, y_target)

            context.stroke()

            # Update so that it is at the top of the next line
            cumulative += count * self.line_width

    def draw_graph(self, filename: str = "output/example.svg",
                   colormap=None,
                   show_timestamps=True, timestamp_translator=None):

        if colormap is None:
            colormap = dict()

        # Instead of passing on dozens of parameters, this checks if the user has already called the necessary functions
        # if not, it is called with the default parameters
        if not self.is_located:
            self.set_locations()

        if show_timestamps:
            if timestamp_translator is not None:
                text_margin = max(map(len, timestamp_translator.values()))*self.font_size
            else:
                text_margin = (math.log10(self.num_layers) + 1)*self.font_size
            self.bottom += text_margin

        surface = cairo.SVGSurface(filename,
                              (self.num_layers * self.xseparation + 2 * self.xmargin) * self.scale,
                              (self.bottom) * self.scale)
        context = cairo.Context(surface)
        context.scale(self.scale, self.scale)

        line_coordinates = dict()  # k: (source, target), v: y-coordinate of endpoint in source.

        for cluster in self.clusters:
            line_coordinates.update(self.calculate_line_origins(cluster))

        already_drawn = set()

        for (source, target) in line_coordinates.keys():
            if (target, source) in already_drawn:
                continue

            self.draw_line(source, target, line_coordinates, colormap, context)
            # self.draw_line_monochrome(source, target, line_coordinates, context)

            already_drawn.add((source, target))
            already_drawn.add((target, source))

        (r, g, b, a) = self.default_cluster_color

        context.set_line_width(self.cluster_width)
        context.set_source_rgba(r, g, b, a)

        for cluster in self.clusters:
            cx, cy = cluster.pos()
            context.move_to(cx, cy - cluster.draw_size / 2.)
            context.line_to(cx, cy + cluster.draw_size / 2.)

        context.stroke()

        if show_timestamps:
            context.set_source_rgb(0, 0, 0)
            context.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            context.set_font_size(self.font_size)  # in user space units # self.xseparation * 0.6

            for t in range(self.num_layers):
                context.move_to(self.xmargin + t * self.xseparation, self.bottom - text_margin)
                context.line_to(self.xmargin + t * self.xseparation, self.bottom - text_margin - self.xseparation * 0.2)

                to_print = str(t)
                if timestamp_translator:
                    to_print = timestamp_translator[t]
                _, _, width, height, _, _ = context.text_extents(to_print)
                context.move_to(self.xmargin + t * self.xseparation + height / 2,
                                self.bottom - text_margin + width + self.xseparation * 0.2)

                context.save()
                context.rotate(math.radians(270))  # angle in rad
                context.show_text(to_print)
                context.restore()

            context.stroke()

