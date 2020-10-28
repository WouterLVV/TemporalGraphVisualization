from collections import Counter

from TimeGraph import TimeGraph, TimeCluster
import cairo
import math

INCOMING = FORWARD = True
OUTGOING = BACKWARD = False


class SugiyamaCluster:
    def __init__(self, tc: TimeCluster, minimum_cluster_size, minimum_connection_size):
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
        self.pos = (-1., -1.)
        self.ypos = -1.
        self.ypos2 = 1.
        self.avg_rank = -1

        # Drawing properties
        self.draw_size = 1.

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

    def __str__(self):
        return f"SugiyamaCluster {str((self.tc.layer, self.tc.id))}/{self.rank} at {str(self.pos)}"

    def __len__(self):
        return len(self.members)


class SugiyamaLayout:

    # ----------------- init Functions --------------------#

    def __init__(self, g: TimeGraph, minimum_cluster_size=0, minimum_connection_size=0):

        # Set basic information from the time graph
        self.g = g
        self.num_layers = self.g.num_steps

        # Set different collection objects
        self.clusters, self.layers = self.build_and_trim_clusters(minimum_cluster_size, minimum_connection_size)
        self.ordered = []  # Filled by reset_order()
        self.reset_order()

        # Set flags
        self.is_ordered = False

        # # This is all drawing parameters so yeet
        # self.scale = 10.
        #
        # self.yseparation = .3
        # self.separation_factor = 1.
        # self.xseparation = 15.0
        #
        #
        # self.xmargin = 10.
        # self.ymargin = 10.
        #
        # self.cluster_height_scale = 1.
        #
        # self.bottom = 0.
        #
        # self.max_chain = -1
        #
        # self.cluster_width = 0.1
        # self.line_width = 0.0
        # self.line_separation = 0.01
        #
        # self.default_line_color = (1., 0., 0., 1.)
        #
        # self.cluster_margin = 0.2

    def build_and_trim_clusters(self, minimum_cluster_size, minimum_connection_size):
        layers = [
                  [cluster for cluster in
                   [SugiyamaCluster(self.g.clusters[t][i], minimum_cluster_size, minimum_connection_size)
                    for i in range(len(self.g.clusters[t]))
                    if len(self.g.clusters[t][i]) >= minimum_cluster_size
                    ]
                   if cluster.outsize + cluster.insize > 0
                   ]
                  for t in range(self.g.num_steps)
                  ]

        clusters = [x for t in range(self.num_layers) for x in layers[t]]

        return clusters, layers

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

    # ---------------- Location Functions ------------------- #

    def set_locations(self):

        for cluster in self.clusters:
            cluster.align = cluster
            cluster.root = cluster
            cluster.draw_size = math.sqrt(len(cluster)) * self.cluster_height_scale
            cluster.ypos = -1.
            cluster.ypos2 = 1.
            cluster.chain_length = 1

        self.align_clusters(direction_flag=True, max_chain=self.max_chain)

        for cluster in [cluster for tc in self.layers for cluster in tc]:
            if cluster.root == cluster:
                self.place_block(cluster)
                # self.place_block_rev(cluster)
        self.bottom = 0.

        # self.avg_positions()
        self.update_positions()

        for _ in range(5):
            for cluster in [cluster for tc in self.layers for cluster in tc]:
                cluster.ypos = -1.

            for cluster in [cluster for tc in self.layers for cluster in tc]:
                if cluster.root == cluster:
                    self.avg_block(cluster)

            self.update_positions()

        self.bottom = max([cluster.pos[1] for tc in self.layers for cluster in tc])
        print("done")
        for (i, order) in enumerate(self.ordered):
            prev = -1
            for cluster in order:
                if cluster.pos[1] <= prev:
                    print("wtf")
                prev = cluster.pos[1]

    def place_block(self, v):
        if v.ypos < 0.:
            v.ypos = 0.
            w = v

            while True:
                if w.rank > 0:
                    u = self.pred(w)
                    self.place_block(u.root)

                    separation_factor = 1.
                    if self.num_shared_neighbours(u, w) > 0:
                        separation_factor = self.separation_factor

                    v.ypos = max(v.ypos,
                                 u.root.ypos + (w.ysize + u.draw_size) / 2. + self.yseparation * separation_factor)

                w = w.align
                if w == v:
                    break

    def place_block_rev(self, v):
        if v.ypos2 > 0.:
            v.ypos2 = 0.
            w = v

            while True:
                u = self.succ(w)
                if u is not None:
                    self.place_block_rev(u.root)

                    separation_factor = 1.
                    if self.num_shared_neighbours(u, w) > 0:
                        separation_factor = self.separation_factor
                    # v.ypos2 = min(v.ypos2, u.root.ypos2 - (self.yseparation + (w.ysize + u.ysize)/2.))
                    v.ypos2 = min(v.ypos2,
                                  u.root.ypos2 - ((w.ysize + u.draw_size) / 2. + self.yseparation * separation_factor))
                w = w.align
                if w == v:
                    break

    def avg_block(self, v):
        if v.ypos < 0.:
            v.ypos = v.pos[1]
            w = v
            upper_bound = v.pos[1]
            lower_bound = float('inf')
            total = 0.
            ctr = 0

            while True:
                u = self.succ(w)
                if u is not None:
                    self.avg_block(u.root)

                    separation_factor = 1.
                    if self.num_shared_neighbours(u, w) > 0:
                        separation_factor = self.separation_factor
                    lower_bound = min(lower_bound, u.root.ypos - self.yseparation * separation_factor - (
                            v.draw_size + u.draw_size) / 2.)
                    # v.ypos = max(v.ypos, u.root.ypos + self.yseparation + (v.ysize + u.ysize)/2.)

                for k, value in w.outgoing.items():
                    if k == w.align:
                        continue
                    ctr += len(value)
                    total += k.pos[1] * len(value)

                for k, value in w.align.incoming.items():
                    if k == w:
                        continue
                    ctr += len(value)
                    total += k.pos[1] * len(value)

                w = w.align
                if w == v:
                    break
            if ctr == 0:
                v.ypos = upper_bound
            else:
                v.ypos = max(upper_bound, min(lower_bound, total / ctr))

    def update_positions(self):

        for cluster in [cluster for tc in self.layers for cluster in tc]:
            cluster.pos = (cluster.tc.layer * self.xseparation + self.xmargin, cluster.root.ypos)

        down = min([cluster.pos[1] for tc in self.layers for cluster in tc]) - self.ymargin
        for cluster in [cluster for tc in self.layers for cluster in tc]:
            cluster.pos = (cluster.pos[0], cluster.pos[1] - down)

    def avg_positions(self):
        down2 = min([cluster.ypos2 for tc in self.layers for cluster in tc])

        for cluster in [cluster for tc in self.layers for cluster in tc]:
            if cluster.root == cluster:
                cluster.ypos2 -= down2

        for cluster in [cluster for tc in self.layers for cluster in tc]:
            cluster.pos = (
                cluster.tc.layer * self.xseparation + self.xmargin, (cluster.root.ypos + cluster.root.ypos2) / 2.)

        down = min([cluster.pos[1] for tc in self.layers for cluster in tc]) - self.ymargin
        for cluster in [cluster for tc in self.layers for cluster in tc]:
            cluster.pos = (cluster.pos[0], cluster.pos[1] - down)

    def expand3(self, csize_func=None):


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
