from collections import Counter

from TimeGraph import TimeGraph, TimeCluster
import cairo
import math

INCOMING = FORWARD = True
OUTGOING = BACKWARD = False


class SugiyamaCluster:
    def __init__(self, tc: TimeCluster, minimum_cluster_size, minimum_connection_size, height_method):
        # Link to original TimeCluster and vice versa
        self.tc = tc
        tc.sc = self

        # Connection properties
        self.incoming = dict()      # Filled by _build()
        self.outgoing = dict()      # Filled by _build()
        self.neighbours = dict()    # Filled by _build()

        self.insize = 0             # Filled by _build()
        self.outsize = 0            # Filled by _build()

        self.members = self.tc.members

        self._build(minimum_cluster_size, minimum_connection_size)

        # Order properties
        self.rank = -1

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

    def _build(self, minimum_cluster_size, minimum_connection_size):
        for c, connection_nodes in self.tc.incoming.items():

            if len(connection_nodes) < minimum_connection_size or len(c) < minimum_cluster_size:
                continue

            self.incoming[c.sc] = connection_nodes
            self.insize += len(connection_nodes)

        for c, connection_nodes in self.tc.incoming.items():

            if len(connection_nodes) < minimum_connection_size or len(c) < minimum_cluster_size:
                continue

            self.incoming[c.sc] = connection_nodes
            self.insize += len(connection_nodes)

        self.neighbours = {**self.incoming, **self.outgoing}

    def reset_alignment(self):
        self.root = self
        self.align = self
        self.chain_length = 1

    def reset_endpoint(self):
        self.align = self.root
        self.root.chain_length = max(1, self.chain_length - 1)

    def pos(self):
        return self.x, self.y

    def __str__(self):
        return f"SugiyamaCluster {str((self.tc.layer, self.tc.id))}/{self.rank} at {str(self.pos)}"

    def __len__(self):
        return len(self.members)


class SugiyamaLayout:

    # ----------------- init Functions --------------------#

    def __init__(self, g: TimeGraph,
                 minimum_cluster_size=0, minimum_connection_size=0,
                 line_width=-1.,
                 horizontal_density=1., vertical_density=1.2,
                 cluster_width=0,
                 cluster_height_method='sqrt',
                 font_size=10):

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

        self.max_cluster_size, self.max_bundle_size = self.calculate_maxes()

        # Location settings
        # 1 point = 0.352 mm, or 3 points = 1 mm
        self.xseparation_frac = horizontal_density  # (fraction) from user
        self.yseparation_frac = vertical_density  # (fraction) from user
        self.line_width = line_width  # (in points) from user
        self.font_size = font_size  # (in points) from user
        self.cluster_width = cluster_width  # (in points) from user; if left 0, computed automatically

        self.scale = 1  # (fraction) do not change
        self.separation_factor = 1  # (fraction) do not change
        self.cluster_margin = 0  # (in points) do not change
        self.xmargin = 20  # (in points) small

        # these are set automatically before drawing
        self.yseparation = 0  # (in points) yseparation_frac * max. bundle thickness, else clusters merge vertically
        self.xseparation = 0  # (in points) xseparation_frac * max. bundle thickness
        self.ymargin = 0  # (in points) 50% of max. bundle thickness, else bottom bundles are cropped halfway
        self.bottom = 0  # (in points) computed automatically from data

        # self.cluster_height_scale = 1 # do not use
        # self.line_separation = 0  # (in points) do not use

        self.max_chain = -1

        self.default_line_color = (0, 0, 0, 1)  # (0, 0.4, 0.8, 1) # r, g, b, a
        self.default_cluster_color = (0, 0, 0)  # r, g, b

    def build_and_trim_clusters(self, minimum_cluster_size, minimum_connection_size, height_method):
        layers = [
                  [cluster for cluster in
                   [SugiyamaCluster(self.g.clusters[t][i], minimum_cluster_size, minimum_connection_size, height_method)
                    for i in range(len(self.g.clusters[t]))
                    if len(self.g.clusters[t][i]) >= minimum_cluster_size
                    ]
                   if cluster.outsize + cluster.insize > 0
                   ]
                  for t in range(self.g.num_steps)
                  ]

        clusters = [x for t in range(self.num_layers) for x in layers[t]]

        return clusters, layers

    def calculate_maxes(self):
        max_cluster_height = max([cluster.draw_size for cluster in self.clusters])
        max_bundle_size = max([max(map(len, cluster.neighbours.values())) for cluster in self.clusters])
        return max_cluster_height, max_bundle_size

    # ---------------- Helper Functions -------------------#

    def pred(self, c: SugiyamaCluster):
        """Returns the predecessor of this cluster (e.g. the cluster with rank-1)
        
        :param c: Cluster to find the predecessor of
        :return: SugiyamaCluster or None if no predecessor exists
        """
        if c.rank == 0:
            return None
        return self.ordered[c.tc.layer][c.rank - 1]

    def succ(self, c: SugiyamaCluster):
        """Returns the successor of this cluster (e.g. the cluster with rank+1)

        :param c: Cluster to find the successor of
        :return: SugiyamaCluster or None if no successor exists
        """
        if c.rank == len(self.layers[c.tc.layer]) - 1:
            return None
        return self.ordered[c.tc.layer][c.rank + 1]

    def largest_median_connection(self, cluster: SugiyamaCluster, lower=True, direction=INCOMING):
        """Returns the cluster with the largest connection to this one.

        If multiple candidates with equal connection weight exist, returns the lower median in ordering

        :param cluster: Cluster to find the median median of
        :param lower: Flag to take either the upper or lower median if even amount
        :param direction: Flag for direction. True is incoming, False is outgoing
        :return: The cluster and the weight of the connection
        """

        if direction:  # is INCOMING
            connections = list(cluster.incoming.items())
        else:  # is OUTGOING
            connections = list(cluster.outgoing.items())

        if len(connections) == 0:
            return None, 0

        connections.sort(key=lambda x: (len(x[1]), x[0].rank), reverse=True)
        ptr = 0
        while ptr < len(connections) and connections[ptr][1] == connections[0][1]:
            ptr += 1

        brother = connections[(ptr - (1 if lower else 0)) // 2][0]
        connsize = len(connections[0][1])
        return brother, connsize

    @staticmethod
    def num_shared_neighbours(u, v):
        return len(set(v.incoming.keys()).intersection(set(u.incoming.keys()))) + len(
            set(v.outgoing.keys()).intersection(set(u.outgoing.keys())))

    # ---------------- Order Functions ------------------- #

    def reset_order(self):
        self.ordered = [self.layers[t].copy() for t in range(self.num_layers)]
        for layer in self.ordered:
            for i, cluster in enumerate(layer):
                cluster.rank = i
        self.is_ordered = False

    def set_order(self, barycenter_passes: int = 20, repetitions_per_pass: int = 5):
        barycenter_passes += (barycenter_passes % 2)

        # Make copy to compare if ordering has stabilized
        orders_tmp = [order.copy() for order in self.ordered]
        direction_flag = FORWARD
        pass_ctr = 0

        # Keep doing passes until the maximum number has been reached or the order does no longer change
        # Always end with a forward pass
        while pass_ctr < barycenter_passes:
            print(pass_ctr)
            self._barycenter(direction_flag, repetitions_per_pass)

            if not direction_flag:
                if orders_tmp == self.ordered:
                    print("Order stabilized")
                    break

                orders_tmp = [order.copy() for order in self.ordered]

            pass_ctr += 1
            direction_flag = not direction_flag

        self.is_ordered = True

    @staticmethod
    def _get_cluster_ranks(cluster):
        if cluster.insize > 0:
            inrank = sum([nb.rank * len(conn) for nb, conn in cluster.incoming.items()]) / cluster.insize
        else:
            inrank = cluster.rank

        if cluster.outsize > 0:
            outrank = sum([n.rank * len(l) for n, l in cluster.outgoing.items()]) / cluster.outsize
        else:
            outrank = cluster.rank

        return inrank, outrank

    def _bary_rank_layer(self, layer, direction_flag):
        # rankmap = {}
        for cluster in layer:
            inr, outr = self._get_cluster_ranks(cluster)

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

    def get_num_crossings(self, cluster1: SugiyamaCluster, cluster2: SugiyamaCluster):
        upper = cluster1
        lower = cluster2
        if upper.rank > lower.rank:
            upper, lower = lower, upper
        return self.get_num_crossings_relative(upper, lower)


    @staticmethod
    def get_num_crossings_relative(upper: SugiyamaCluster, lower: SugiyamaCluster):
        sins_lower = sorted(upper.incoming.items(), key=lambda x: x[0].rank)
        sins_upper = sorted(lower.incoming.items(), key=lambda x: x[0].rank)
        i = j = crossings = 0

        while i < len(sins_lower) and j < len(sins_upper):
            if sins_upper[j] > sins_lower[i]:
                crossings += len(sins_lower) - i
                i += 1
            elif sins_upper[j] < sins_lower[i]:
                j += 1
            else:
                i += 1
                j += 1
        return crossings

    def crossing_diff_if_swapped(self, cluster1: SugiyamaCluster, cluster2: SugiyamaCluster):
        upper = cluster1
        lower = cluster2
        if upper.rank > lower.rank:
            upper, lower = lower, upper
        return self.get_num_crossings_relative(lower, upper) - self.get_num_crossings_relative(upper, lower)

    def swap_clusters(self, cluster1: SugiyamaCluster, cluster2: SugiyamaCluster, return_crossing_diff=False):
        if return_crossing_diff:
            crossings = self.get_num_crossings(cluster1, cluster2)

        self.ordered[cluster1.rank], self.ordered[cluster2.rank] = self.ordered[cluster2.rank], self.ordered[cluster1.rank]
        cluster1.rank, cluster2.rank = cluster2.rank, cluster1.rank

        if return_crossing_diff:
            return self.get_num_crossings(cluster1, cluster2) - crossings

    # ---------------- Alignment Functions ------------------ #

    def reset_alignments(self):
        for cluster in self.clusters:
            cluster.reset_alignment()

    def has_larger_crossings(self, start_cluster, until_rank, connection_size):
        """Checks for crossing of at least a certain size until it is found or a certain rank is reached

        :param start_cluster: First cluster to check connections from. Should have a lower rank than until_rank
        :param until_rank: Continue up to and including the cluster of this rank.
        :param connection_size: Threshold for which to check.
        """
        cluster = start_cluster
        while cluster.rank <= until_rank:
            if cluster.align != cluster.root and len(cluster.neighbours[cluster.align]) > connection_size:
                return True
            cluster = self.pred(cluster)
        return False

    def remove_crossings(self, start_cluster, until_rank):
        """Removes all alignments of clusters from start_cluster until a certain rank is reached

        :param start_cluster: first cluster to remove alignment from. Should have a lower rank than until_rank
        :param until_rank: continue up to and including the cluster of this rank.
        """
        cluster = start_cluster
        while cluster.rank <= until_rank:
            if cluster.align != cluster.root:  # cluster must not be an endpoint
                cluster.align.reset_alignment()
                cluster.reset_endpoint()
            cluster = self.pred(cluster)

    @staticmethod
    def align_cluster(endpoint: SugiyamaCluster, new: SugiyamaCluster):
        endpoint.align = new
        new.root = endpoint.root
        new.align = endpoint.root
        endpoint.root.chain_length += 1

    def align_clusters(self, direction_flag=FORWARD, max_chain=-1):
        if not self.is_ordered:
            self.set_order()

        if direction_flag:
            layer_range = range(1, self.g.num_steps)
        else:
            layer_range = range(self.g.num_steps - 2, -1, -1)

        for layers in layer_range:
            r = -1

            for cluster in self.ordered[layers]:
                # Find cluster in previous layer this one wants to connect to and the weight of the connection
                brother, connsize = self.largest_median_connection(cluster, direction=direction_flag)

                if brother is not None and (max_chain < 0 or brother.root.chain_length <= max_chain):

                    # Check if this connection contradicts another alignment
                    # priority to the new connection is only given if the weight is higher than all crossings
                    if brother.rank <= r:
                        if self.has_larger_crossings(self.pred(brother), r, connsize):
                            continue
                        self.remove_crossings(self.pred(brother), r)

                    self.align_cluster(brother, cluster)
                    r = brother.rank

    def crossing_diff_if_swapped_align(self, upper: SugiyamaCluster, lower: SugiyamaCluster):
        uroot = upper.root
        lroot = lower.root
        cluster = lroot
        crossing_diff = 0
        while True:
            predecessor = self.pred(cluster)
            if predecessor.root == uroot:
                crossing_diff += self.crossing_diff_if_swapped(cluster, predecessor)

            cluster = cluster.align
            if cluster == lroot:
                break
        return crossing_diff

    def swap_align(self, upper: SugiyamaCluster, lower: SugiyamaCluster):
        uroot = upper.root
        lroot = lower.root
        cluster = lroot
        while True:
            predecessor = self.pred(cluster)
            if predecessor.root == uroot:
                self.swap_clusters(cluster, predecessor)

            cluster = cluster.align
            if cluster == lroot:
                break

    def wanted_direction(self, root: SugiyamaCluster):
        l = []
        l.append(self.largest_median_connection(root, direction=INCOMING))
        cluster = root
        while True:
            l.append(cluster)
            cluster = cluster.align
            if cluster.align == root:
                break

        l.append(self.largest_median_connection(cluster, direction=OUTGOING))

        total = 0
        for i in range(1, len(l) - 1):
            cluster = l[i]
            for k, v in cluster.incoming.items():
                total += len(v)*(k.rank - l[i-1].rank)

            for k, v in cluster.outgoing.items():
                total += len(v)*(k.rank - l[i+1].rank)

        return total

    def collapse_stairs_iteration(self):
        for cluster in self.clusters:
            if cluster.root != cluster:
                continue

            cluster.wanted_direction = self.wanted_direction(cluster)

        for cluster in self.clusters:
            if cluster.root != cluster:
                continue

            if cluster.wanted_direction > 0:

                successor = self.succ(cluster)
                while successor is not None and successor.root.chain_length > cluster.chain_length and self.crossing_diff_if_swapped_align(
                        cluster, successor) < 1:
                    self.swap_align(cluster, successor)
                    successor = self.succ(cluster)

                predecessor = self.pred(cluster)
                while predecessor is not None and predecessor.root.chain_length < cluster.chain_length and predecessor.root.wanted_direction > 0 and self.crossing_diff_if_swapped_align(predecessor, cluster) < 1:
                    self.swap_align(self.pred(cluster), cluster)
                    predecessor = self.pred(cluster)

            elif cluster.wanted_direction < 0:
                predecessor = self.pred(cluster)
                while predecessor is not None and predecessor.root.chain_length < cluster.chain_length and self.crossing_diff_if_swapped_align(predecessor, cluster) < 1:
                    self.swap_align(self.pred(cluster), cluster)
                    predecessor = self.pred(cluster)

                successor = self.succ(cluster)
                while successor is not None and successor.root.chain_length > cluster.chain_length and predecessor.root.wanted_direction < 0 and self.crossing_diff_if_swapped_align(cluster, successor) < 1:
                    self.swap_align(cluster, successor)
                    successor = self.succ(cluster)

    # ---------------- Location Functions ------------------- #

    def set_locations(self, cluster_separation=-1.):

        if not self.is_aligned:
            self.align_clusters()

        for cluster in self.clusters:
            if cluster.root == cluster:
                self.place_block(cluster)

        self.update_positions()

        for _ in range(5):
            for cluster in self.clusters:
                cluster._y = -1.

            for cluster in self.clusters:
                if cluster.root == cluster:
                    self.avg_block(cluster)

            self.update_positions()

        self.check_locations()
        print("done")

    def check_locations(self):
        for (i, order) in enumerate(self.ordered):
            prev = -1
            for cluster in order:
                if cluster.pos[1] <= prev:
                    print("wtf")
                prev = cluster.pos[1]

    def center_distance(self, u, v):
        return self.yseparation_frac * (u.draw_size + v.draw_size) / 2.

    def place_block(self, root):
        if root._y < 0.:
            root._y = 0.
            cluster = root

            while True:
                if cluster.rank > 0:
                    predecessor = self.pred(cluster)
                    self.place_block(predecessor.root)

                    root.y = max(root._y, predecessor.root._y + self.center_distance(predecessor, cluster))

                cluster = cluster.align
                if cluster == root:
                    break

    def avg_block(self, root):
        if root._y < 0.:
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

    def update_positions(self):
        for cluster in self.clusters:
            cluster.y = cluster._y

    # ---------------- Drawing Functions ------------------- #

    def _fit_line_width(self):
        max_line_width = float('inf')
        for cluster in self.clusters:
            line_width_in = line_width_out = float('inf')
            if cluster.insize > 0:
                line_width_in = (cluster.draw_size - (
                        len(cluster.incoming.keys()) - 1) * self.line_separation) / cluster.insize
            if cluster.outsize > 0:
                line_width_out = (cluster.draw_size - (
                        len(cluster.outgoing.keys()) - 1) * self.line_separation) / cluster.outsize
            max_line_width = min(max_line_width, line_width_in, line_width_out)
        return max_line_width * (1. - 2 * self.cluster_margin)

    def draw_graph(self, filename: str = "output/example.svg", ignore_loners=False, marked_nodes=None,
                   max_iterations=100, colormap=None):
        if colormap is None:
            colormap = dict()

        if marked_nodes is None:
            marked_nodes = set()
        self.set_locations(ignore_loners, max_pass=max_iterations, reps=10)

        with cairo.SVGSurface(filename,
                              (self.g.num_steps) * self.xseparation * self.scale + 2 * self.xmargin,
                              self.bottom * self.scale + 2 * self.ymargin) as surface:
            context = cairo.Context(surface)
            context.scale(self.scale, self.scale)

            self.line_width = self._fit_line_width()

            context.set_source_rgb(1., 0., 0.)
            context.set_line_width(self.line_width)
            rel_line_coords = dict()
            for t in range(self.g.num_steps):
                for cluster in self.layers[t]:
                    incomings = list(cluster.incoming.items())
                    incomings.sort(key=lambda x: x[0].rank)
                    outgoings = list(cluster.outgoing.items())
                    outgoings.sort(key=lambda x: x[0].rank)

                    ctr = 0
                    for (c, members) in incomings:
                        width = len(members)
                        rel_line_coords[(cluster, c)] = (ctr + width / 2) / cluster.insize
                        ctr += width

                    ctr = 0
                    for (c, members) in outgoings:
                        width = len(members)
                        rel_line_coords[(cluster, c)] = (ctr + width / 2) / cluster.outsize
                        ctr += width

            for t in range(self.g.num_steps - 1):
                for cluster in self.layers[t]:

                    stop = cluster.pos[1] - cluster.draw_size / 2.
                    outgoings = list(cluster.outgoing.items())

                    for (target, members) in outgoings:
                        weight = len(members)

                        ttop = target.pos[1] - target.draw_size / 2.

                        source_y = stop + cluster.draw_size * self.cluster_margin + cluster.draw_size * (
                                1 - 2 * self.cluster_margin) * rel_line_coords[(cluster, target)]
                        target_y = ttop + target.draw_size * self.cluster_margin + target.draw_size * (
                                1 - 2 * self.cluster_margin) * rel_line_coords[(target, cluster)]

                        lines = dict()
                        for mem in members:
                            if mem.name not in lines.keys():
                                lines[mem.name] = 0
                            lines[mem.name] += 1

                        ctr = 0
                        for name, thiccness in sorted(list(lines.items())):
                            ctr += thiccness / 2.
                            context.set_line_width(self.line_width * thiccness)
                            if name in colormap.keys():
                                (r, g, b, a) = colormap[name]
                            else:
                                (r, g, b, a) = self.default_line_color
                            context.set_source_rgba(r, g, b, a)

                            y_start = source_y + (ctr - weight / 2) * self.line_width
                            y_end = target_y + (ctr - weight / 2) * self.line_width

                            context.move_to(cluster.pos[0], y_start)
                            context.curve_to(cluster.pos[0] + self.xseparation * 0.3, y_start,
                                             target.pos[0] - self.xseparation * 0.3, y_end, target.pos[0], y_end)

                            context.stroke()
                            ctr += thiccness / 2.

            context.stroke()

            context.set_line_width(self.cluster_width)
            context.set_source_rgb(0., 0., 0.)

            for t in range(self.g.num_steps):
                for cluster in self.layers[t]:
                    cx, cy = cluster.pos
                    context.move_to(cx, cy - cluster.draw_size / 2.)
                    context.line_to(cx, cy + cluster.draw_size / 2.)

            context.stroke()
