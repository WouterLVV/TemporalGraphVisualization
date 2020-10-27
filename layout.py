from collections import Counter

from TimeGraph import TimeGraph, TimeCluster
import cairo
import math




class SugiyamaCluster:
    def __init__(self, tc: TimeCluster):
        self.tc = tc
        tc.sc = self

        self.rank = self.tc.id
        self.avg_rank = self.rank
        self.root = self
        self.align = self
        self.ys = []
        self.chain_length = -1

        self.ysize = 1.

        self.pos = (-1., -1.)
        self.ypos = -1.
        self.ypos2 = 1.

        self.members = self.tc.members

        self.incoming = dict()
        self.outgoing = dict()

        self.insize = 0
        self.outsize = 0

    def __str__(self):
        return f"SugiyamaCluster {str((self.tc.layer, self.tc.id))}/{self.rank} at {str(self.pos)}"

    def __len__(self):
        return len(self.members)


class SugiyamaLayout:
    def __init__(self, g: TimeGraph, min_clust_size=0, min_conn_size=0):
        self.g = g

        self.num_layers = self.g.num_steps

        self.layers = [
                [SugiyamaCluster(self.g.clusters[t][i])
                    for i in range(len(self.g.clusters[t]))
                    if len(self.g.clusters[t][i]) >= min_clust_size]
            for t in range(self.g.num_steps)]

        # flatten layers for easy looping
        self.clusters = [x for t in range(self.g.num_steps) for x in self.layers[t]]


        self.max_conn_size = 0
        self.max_clust_ysize = 0.

        for cluster in self.clusters:
            for c, mems in cluster.tc.incoming.items():
                if len(mems) >= min_conn_size and len(c) >= min_clust_size:
                    cluster.incoming[c.sc] = mems
                    cluster.insize += len(mems)

                    c.sc.outgoing[cluster] = mems
                    c.sc.outsize += len(mems)

                    self.max_conn_size = max(self.max_conn_size, len(mems))

        to_remove = set()
        for t in range(self.g.num_steps):
            for cluster in self.layers[t]:
                if cluster.outsize + cluster.insize == 0:
                    to_remove.add(cluster)

        for cluster in to_remove:
            self.layers[cluster.tc.layer].remove(cluster)
            self.clusters.remove(cluster)

        self.ordered = [self.layers[t].copy() for t in range(self.g.num_steps)]



        self.scale = 10.

        self.yseparation = .3
        self.separation_factor = 1.
        self.xseparation = 15.0


        self.xmargin = 10.
        self.ymargin = 10.

        self.cluster_height_scale = 1.

        self.bottom = 0.

        self.max_chain = -1

        self.cluster_width = 0.1
        self.line_width = 0.0
        self.line_separation = 0.01

        self.default_line_color = (1., 0., 0., 1.)

        self.cluster_margin = 0.2

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
        """Returns the predecessor of this cluster (e.g. the cluster with rank+1)

        :param c: Cluster to find the predecessor of
        :return: SugiyamaCluster or None if no predecessor exists
        """
        if c.rank == len(self.layers[c.tc.layer]) - 1:
            return None
        return self.ordered[c.tc.layer][c.rank + 1]

    def largest_median_connection(self, c: SugiyamaCluster, lower=True, incoming=True):
        """Returns the cluster with the largest connection to this one.

        If multiple candidates with equal connection weight exist, returns the lower median in ordering

        :param c: Cluster to find the incoming median of
        :return: The cluster and the weight of the connection
        """

        if incoming:
            connections = list(c.incoming.items())
        else:
            connections = list(c.outgoing.items())

        if len(connections) == 0:
            return None, 0

        connections.sort(key=lambda x: (len(x[1]), x[0].rank), reverse=True)
        ptr = 0
        while ptr < len(connections) and connections[ptr][1] == connections[0][1]:
            ptr += 1

        brother = connections[(ptr - (1 if lower else 0)) // 2][0]
        connsize = len(connections[0][1])
        return brother, connsize

    # ---------------- Location Functions -------------------#

    def set_locations(self, ignore_loners=False, max_pass=20, reps = 5):
        max_pass += (max_pass % 2)
        # First layer is initialized on their id
        for i, cluster in enumerate(self.ordered[0]):
            cluster.rank = i

        orders_tmp = [order.copy() for order in self.ordered]
        dir_switch = True
        pass_ctr = 0

        # Keep doing passes until the maximum number has been reached or the order does no longer change
        # Always end with a forward pass
        while pass_ctr < max_pass:
            self._barycenter(dir_switch, reps)

            if not dir_switch:
                if orders_tmp == self.ordered:
                    break

                orders_tmp = [order.copy() for order in self.ordered]

            pass_ctr += 1
            dir_switch = not dir_switch
            print(pass_ctr)


        self.expand3()



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

    def _bary_rank_layer(self, layer, forward):
        # rankmap = {}
        for cluster in layer:
            inr, outr = self._get_cluster_ranks(cluster)

            if forward:
                cluster.rank = inr
                cluster.secondrank = outr
            else:
                cluster.rank = outr
                cluster.secondrank = inr

        layer.sort(key=lambda c: (c.rank, c.secondrank))
        for i, cluster in enumerate(layer):
            cluster.rank = i


    def _barycenter(self, dir_switch, reps):

        # True means forward pass (leaves first layer unchanged)
        # False is backwards pass (last layer unchanged)
        if dir_switch:
            r = range(1, self.num_layers)
        else:
            r = range(self.num_layers - 2, -1, -1)

        for _ in range(reps):
            for t in r:
                layer = self.ordered[t]
                self._bary_rank_layer(layer, dir_switch)

    def shared_neighbours(self, u, v):
        return len(set(v.incoming.keys()).intersection(set(u.incoming.keys()))) + len(set(v.outgoing.keys()).intersection(set(u.outgoing.keys())))

    def place_block(self, v):
        if v.ypos < 0.:
            v.ypos = 0.
            w = v

            while True:
                if w.rank > 0:
                    u = self.pred(w)
                    self.place_block(u.root)

                    # # Sink stuff does not work properly
                    # if v.sink == v:
                    #     v.sink = u.sink
                    # if v.sink != u.sink:
                    #     u.sink.shift = min(u.sink.shift, v.ypos - u.ypos - self.yseparation)
                    # else:
                    # v.ypos = max(v.ypos, u.root.ypos + self.yseparation + (w.ysize + u.ysize)/2.)
                    separation_factor = 1.
                    if self.shared_neighbours(u, w) > 0:
                        separation_factor = self.separation_factor

                    v.ypos = max(v.ypos, u.root.ypos + (w.ysize + u.ysize) / 2. + self.yseparation * separation_factor)


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

                    # # Sink stuff does not work properly
                    # if v.sink == v:
                    #     v.sink = u.sink
                    # if v.sink != u.sink:
                    #     u.sink.shift = min(u.sink.shift, v.ypos - u.ypos - self.yseparation)
                    # else:

                    separation_factor = 1.
                    if self.shared_neighbours(u, w) > 0:
                        separation_factor = self.separation_factor
                    # v.ypos2 = min(v.ypos2, u.root.ypos2 - (self.yseparation + (w.ysize + u.ysize)/2.))
                    v.ypos2 = min(v.ypos2, u.root.ypos2 - ((w.ysize + u.ysize) / 2. + self.yseparation * separation_factor))
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

                    # # Sink stuff does not work properly
                    # if v.sink == v:
                    #     v.sink = u.sink
                    # if v.sink != u.sink:
                    #     u.sink.shift = min(u.sink.shift, v.ypos - u.ypos - self.yseparation)
                    # else:
                    separation_factor = 1.
                    if self.shared_neighbours(u, w) > 0:
                        separation_factor = self.separation_factor
                    lower_bound = min(lower_bound, u.root.ypos - self.yseparation*separation_factor - (v.ysize + u.ysize) / 2.)
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

    def align_clusters(self, forward=True, max_chain=-1):

        if forward:
            it = range(1, self.g.num_steps)
            l = 1
        else:
            it = range(self.g.num_steps - 2, -1, -1)
            l = -1

        for t in it:
            r = -1

            for cluster in self.ordered[t]:
                # Find cluster in previous layer this one wants to connect to and the weight of the connection
                brother, connsize = self.largest_median_connection(cluster, incoming=forward)

                if cluster.align == cluster and brother is not None:

                    allowed = max_chain < 0 or brother.root.chain_length <= max_chain


                    # Check if this connection contradicts another alignment
                    # priority to the new connection is only given if the weight is higher than all crossings
                    if brother.rank <= r:
                        for i in range(brother.rank, r + 1):
                            prev_cluster = self.ordered[t - l][i]
                            if prev_cluster.align != prev_cluster.root and \
                                    len((prev_cluster.outgoing if forward else prev_cluster.incoming)[
                                        prev_cluster.align]) > connsize:
                                allowed = False
                                break

                        # If the new connection is allowed to exist, first remove all current connections that cross it
                        if allowed:
                            for i in range(brother.rank, r + 1):
                                prev_cluster = self.ordered[t - l][i]
                                if prev_cluster.align != prev_cluster.root:  # prev_cluster must not be an endpoint
                                    node_to_reset = prev_cluster.align
                                    node_to_reset.align = node_to_reset
                                    node_to_reset.root = node_to_reset
                                    node_to_reset.chain_length = 1
                                    prev_cluster.align = prev_cluster.root
                                    prev_cluster.root.chain_length -= 1

                    if allowed:
                        brother.align = cluster
                        cluster.root = brother.root
                        cluster.align = cluster.root
                        cluster.root.chain_length += 1

                        r = brother.rank



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
        if csize_func is None:
            csize_func = lambda c: math.sqrt(len(c))

        for cluster in self.clusters:
            cluster.align = cluster
            cluster.root = cluster
            cluster.ysize = csize_func(cluster) * self.cluster_height_scale
            cluster.ypos = -1.
            cluster.ypos2 = 1.
            cluster.chain_length = 1

        self.align_clusters(forward=True, max_chain=self.max_chain)

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

    # ---------------- Drawing Functions ------------------- #

    def _fit_line_width(self):
        max_line_width = float('inf')
        for cluster in self.clusters:
            line_width_in = line_width_out = float('inf')
            if cluster.insize > 0:
                line_width_in = (cluster.ysize - (len(cluster.incoming.keys()) - 1)*self.line_separation) / cluster.insize
            if cluster.outsize > 0:
                line_width_out = (cluster.ysize - (len(cluster.outgoing.keys()) - 1) * self.line_separation) / cluster.outsize
            max_line_width = min(max_line_width, line_width_in, line_width_out)
        return max_line_width*(1.-2*self.cluster_margin)

    def draw_graph(self, filename: str = "output/example.svg", ignore_loners=False, marked_nodes=None, max_iterations=100, colormap=None):
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

                    stop = cluster.pos[1] - cluster.ysize / 2.
                    outgoings = list(cluster.outgoing.items())

                    for (target, members) in outgoings:
                        weight = len(members)

                        ttop = target.pos[1] - target.ysize / 2.

                        source_y = stop + cluster.ysize * self.cluster_margin + cluster.ysize * (
                                1 - 2 * self.cluster_margin) * rel_line_coords[(cluster, target)]
                        target_y = ttop + target.ysize * self.cluster_margin + target.ysize * (
                                1 - 2 * self.cluster_margin) * rel_line_coords[(target, cluster)]

                        lines = dict()
                        for mem in members:
                            if mem.name not in lines.keys():
                                lines[mem.name] = 0
                            lines[mem.name] += 1

                        ctr = 0
                        for name, thiccness in sorted(list(lines.items())):
                            ctr += thiccness/2.
                            context.set_line_width(self.line_width*thiccness)
                            if name in colormap.keys():
                                (r,g,b,a) = colormap[name]
                            else:
                                (r,g,b,a) = self.default_line_color
                            context.set_source_rgba(r, g, b, a)



                            y_start = source_y + (ctr - weight/2)*self.line_width
                            y_end = target_y + (ctr - weight/2)*self.line_width

                            context.move_to(cluster.pos[0], y_start)
                            context.curve_to(cluster.pos[0] + self.xseparation * 0.3, y_start,
                                             target.pos[0] - self.xseparation * 0.3, y_end, target.pos[0], y_end)

                            context.stroke()
                            ctr += thiccness/2.

            context.stroke()

            context.set_line_width(self.cluster_width)
            context.set_source_rgb(0., 0., 0.)

            for t in range(self.g.num_steps):
                for cluster in self.layers[t]:
                    cx, cy = cluster.pos
                    context.move_to(cx, cy - cluster.ysize / 2.)
                    context.line_to(cx, cy + cluster.ysize / 2.)

            context.stroke()
