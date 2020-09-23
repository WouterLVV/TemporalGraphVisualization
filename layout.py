from TimeGraph import TimeGraph, TimeCluster
import cairo


class SugiyamaCluster:
    def __init__(self, tc: TimeCluster):
        self.tc = tc
        tc.sc = self

        self.rank = self.tc.id
        self.avg_rank = self.rank
        self.root = self
        self.align = self
        self.ys = []
        self.chain_length = 1

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


class SugiyamaLayout:
    def __init__(self, g: TimeGraph, minimum_cluster_size=0, minimum_connections_size=0):
        self.g = g

        self.clusters = [
            [SugiyamaCluster(self.g.clusters[t][i]) for i in range(len(self.g.clusters[t]))
             if len(self.g.clusters[t][i]) >= minimum_cluster_size]
            for t in range(self.g.num_steps)]

        for t in range(self.g.num_steps):
            for cluster in self.clusters[t]:
                for c, mems in cluster.tc.incoming.items():
                    if len(mems) >= minimum_connections_size and len(c) >= minimum_cluster_size:
                        cluster.incoming[c.sc] = len(mems)
                        cluster.insize += len(mems)

                        c.sc.outgoing[cluster] = len(mems)
                        c.sc.outsize += len(mems)

        toremove = set()
        for t in range(self.g.num_steps):
            for cluster in self.clusters[t]:
                if cluster.outsize + cluster.insize == 0:
                    toremove.add(cluster)

        for cluster in toremove:
            self.clusters[cluster.tc.layer].remove(cluster)

        self.orders = [self.clusters[t].copy() for t in range(self.g.num_steps)]

        self.num_layers = self.g.num_steps

        self.scale = 2.
        self.yseparation = 5.0
        self.xseparation = 15.0

        self.xmargin = 1.
        self.ymargin = 1.

        self.cluster_height_scale = 0.

        self.bottom = 0.

        self.max_chain = -1

        self.cluster_width = 2.
        self.line_width = 0.2
        self.line_separation = 0.1

        self.cluster_margin = 0.5 - (self.line_width + self.line_separation) / 2.

    # ---------------- Helper Functions -------------------#

    def pred(self, c: SugiyamaCluster):
        """Returns the predecessor of this cluster (e.g. the cluster with rank-1)
        
        :param c: Cluster to find the predecessor of
        :return: SugiyamaCluster or None if no predecessor exists
        """
        if c.rank == 0:
            return None
        return self.orders[c.tc.layer][c.rank - 1]

    def succ(self, c: SugiyamaCluster):
        """Returns the predecessor of this cluster (e.g. the cluster with rank+1)

        :param c: Cluster to find the predecessor of
        :return: SugiyamaCluster or None if no predecessor exists
        """
        if c.rank == len(self.clusters[c.tc.layer]) - 1:
            return None
        return self.orders[c.tc.layer][c.rank + 1]

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

        connections.sort(key=lambda x: (x[1], x[0].rank), reverse=True)
        ptr = 0
        while ptr < len(connections) and connections[ptr][1] == connections[0][1]:
            ptr += 1

        brother = connections[(ptr - (1 if lower else 0)) // 2][0]
        connsize = connections[0][1]
        return brother, connsize

    # ---------------- Location Functions -------------------#

    def set_locations(self, ignore_loners=False, max_pass=100):

        # First layer is initialized on their id
        for i, cluster in enumerate(self.clusters[0]):
            cluster.pos = (self.xmargin, i)
            cluster.rank = i

        orders = [self.clusters[t].copy() for t in range(self.g.num_steps)]
        orders_tmp = [order.copy() for order in orders]
        forward = True
        pass_ctr = 0

        # Keep doing passes until the maximum number has been reached or the order does no longer change
        # Always end with a forward pass
        while pass_ctr < max_pass or forward:
            orders = self.barycenter_pass(forward, orders)

            if forward:
                unchanged = True
                for t in range(self.num_layers - 1):
                    unchanged &= orders[t] == orders_tmp[t]
                    if not unchanged:
                        break
                if unchanged:
                    break
                orders_tmp = [order.copy() for order in orders]

            pass_ctr += 1
            forward = not forward
            print(pass_ctr)
        # for order in orders:
        #     for i, cluster in enumerate(order):
        #         cluster.rank = i
        orders = self.barycenter_pass(forward=True, orders=orders)
        orders = self.barycenter_pass(forward=True, orders=orders)
        orders = self.barycenter_pass(forward=True, orders=orders)
        orders = self.barycenter_pass(forward=True, orders=orders)
        self.orders = orders
        self.expand3()

    def barycenter_pass(self, forward, orders, alpha = 0.95):
        if forward:
            r = range(1, self.num_layers)
        else:
            r = range(self.num_layers - 2, -1, -1)

        for t in r:
            order = orders[t]
            for cluster in order:
                inrank = 0.
                outrank = 0.
                if cluster.insize > 0:
                    inrank = sum([n.rank * l for n, l in cluster.incoming.items()]) / cluster.insize
                else:
                    inrank = cluster.rank

                if cluster.outsize > 0:
                    outrank = sum([n.rank * l for n, l in cluster.outgoing.items()]) / cluster.outsize
                else:
                    outrank = cluster.rank

                if forward:
                    cluster.rank = inrank
                    cluster.secondrank = outrank
                else:
                    cluster.rank = outrank
                    cluster.secondrank = inrank

                # if forward:
                #     if cluster.insize > 0:
                #         cluster.rank = ((alpha * sum([n.rank * l for n, l in cluster.incoming.items()])) / cluster.insize) + ((1.-alpha)*cluster.rank)
                # else:
                #     if cluster.outsize > 0:
                #         cluster.rank = ((alpha * sum([n.rank * l for n, l in cluster.outgoing.items()])) / cluster.outsize) + ((1.-alpha)*cluster.rank)

            if t == self.num_layers - 30:
                print("yeet")
            order.sort(key=lambda c: (c.rank, c.secondrank))
            for i, cluster in enumerate(order):
                cluster.rank = i

        return orders

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
                    v.ypos = max(v.ypos, u.root.ypos + (w.ysize + u.ysize) / 2. + self.yseparation * (1. if (len(set(w.incoming.keys()).intersection(set(u.incoming.keys()))) + len(set(w.outgoing.keys()).intersection(set(u.outgoing.keys()))) > 0) else 4.))


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

                    # v.ypos2 = min(v.ypos2, u.root.ypos2 - (self.yseparation + (w.ysize + u.ysize)/2.))
                    v.ypos2 = min(v.ypos2, u.root.ypos2 - ((w.ysize + u.ysize) / 2. + self.yseparation * (1. if (len(set(w.incoming.keys()).intersection(set(u.incoming.keys()))) + len(set(w.outgoing.keys()).intersection(set(u.outgoing.keys()))) > 0) else 4.)))
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
                    lower_bound = min(lower_bound, u.root.ypos - self.yseparation - (v.ysize + u.ysize) / 2.)
                    # v.ypos = max(v.ypos, u.root.ypos + self.yseparation + (v.ysize + u.ysize)/2.)

                for k, value in w.outgoing.items():
                    if k == w.align:
                        continue
                    ctr += value
                    total += k.pos[1] * value

                for k, value in w.align.incoming.items():
                    if k == w:
                        continue
                    ctr += value
                    total += k.pos[1] * value

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

            for cluster in self.orders[t]:
                # Find cluster in previous layer this one wants to connect to and the weight of the connection
                brother, connsize = self.largest_median_connection(cluster, incoming=forward)

                if cluster.align == cluster and brother is not None:

                    allowed = max_chain < 0 or brother.root.chain_length <= max_chain


                    # Check if this connection contradicts another alignment
                    # priority to the new connection is only given if the weight is higher than all crossings
                    if brother.rank <= r:
                        for i in range(brother.rank, r + 1):
                            prev_cluster = self.orders[t - l][i]
                            if prev_cluster.align != prev_cluster.root and \
                                    (prev_cluster.outgoing if forward else prev_cluster.incoming)[
                                        prev_cluster.align] > connsize:
                                allowed = False
                                break

                        # If the new connection is allowed to exist, first remove all current connections that cross it
                        if allowed:
                            for i in range(brother.rank, r + 1):
                                prev_cluster = self.orders[t - l][i]
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

        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            cluster.pos = (cluster.tc.layer * self.xseparation + self.xmargin, cluster.root.ypos)

        down = min([cluster.pos[1] for tc in self.clusters for cluster in tc]) - self.ymargin
        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            cluster.pos = (cluster.pos[0], cluster.pos[1] - down)

    def avg_positions(self):
        down2 = min([cluster.ypos2 for tc in self.clusters for cluster in tc])

        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            if cluster.root == cluster:
                cluster.ypos2 -= down2

        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            cluster.pos = (
            cluster.tc.layer * self.xseparation + self.xmargin, (cluster.root.ypos + cluster.root.ypos2) / 2.)

        down = min([cluster.pos[1] for tc in self.clusters for cluster in tc]) - self.ymargin
        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            cluster.pos = (cluster.pos[0], cluster.pos[1] - down)

    def expand3(self):
        inf = float('inf')
        for tc in self.clusters:
            for cluster in tc:
                cluster.align = cluster
                cluster.root = cluster
                # cluster.sink = cluster
                # cluster.shift = inf
                cluster.ysize = len(cluster.tc) * self.cluster_height_scale
                cluster.ypos = -1.
                cluster.ypos2 = 1.
                cluster.chain_length = 1

        self.align_clusters(forward=True, max_chain=self.max_chain)

        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            if cluster.root == cluster:
                self.place_block(cluster)
                # self.place_block_rev(cluster)
        self.bottom = 0.

        self.avg_positions()

        for _ in range(5):
            for cluster in [cluster for tc in self.clusters for cluster in tc]:
                cluster.ypos = -1.

            for cluster in [cluster for tc in self.clusters for cluster in tc]:
                if cluster.root == cluster:
                    self.avg_block(cluster)

            self.update_positions()

        self.bottom = max([cluster.pos[1] for tc in self.clusters for cluster in tc])
        print("done")
        for (i, order) in enumerate(self.orders):
            prev = -1
            for cluster in order:
                if cluster.pos[1] <= prev:
                    print("wtf")
                prev = cluster.pos[1]

    # ---------------- Drawing Functions ------------------- #

    def draw_graph(self, filename: str = "output/example.svg", ignore_loners=False, marked_nodes=None, max_iterations=100):
        if marked_nodes is None:
            marked_nodes = set()
        self.set_locations(ignore_loners, max_pass=max_iterations)

        with cairo.SVGSurface(filename,
                              (self.g.num_steps) * self.xseparation * self.scale + 2 * self.xmargin,
                              self.bottom * self.scale + 2 * self.ymargin) as surface:
            context = cairo.Context(surface)
            context.scale(self.scale, self.scale)

            context.set_source_rgb(1., 0., 0.)
            context.set_line_width(self.line_width)
            rel_line_coords = dict()
            for t in range(self.g.num_steps):
                for cluster in self.clusters[t]:
                    incomings = list(cluster.incoming.items())
                    incomings.sort(key=lambda x: x[0].rank)
                    outgoings = list(cluster.outgoing.items())
                    outgoings.sort(key=lambda x: x[0].rank)

                    ctr = 0
                    for (c, members) in incomings:
                        width = members
                        rel_line_coords[(cluster, c)] = (ctr + width / 2) / cluster.insize
                        ctr += width

                    ctr = 0
                    for (c, members) in outgoings:
                        width = members
                        rel_line_coords[(cluster, c)] = (ctr + width / 2) / cluster.outsize
                        ctr += width

            for t in range(self.g.num_steps - 1):
                for cluster in self.clusters[t]:

                    stop = cluster.pos[1] - cluster.ysize / 2.
                    outgoings = list(cluster.outgoing.items())

                    for (target, members) in outgoings:

                        weight = members
                        context.set_line_width(self.line_width * weight)
                        if len(marked_nodes.intersection(cluster.tc.outgoing[target.tc])) > 0:
                            context.set_source_rgb(0., 0., 1.)
                        else:
                            context.set_source_rgb(1., 0., 0.)

                        ttop = target.pos[1] - target.ysize / 2.

                        source_y = stop + cluster.ysize * self.cluster_margin + cluster.ysize * (
                                    1 - 2 * self.cluster_margin) * rel_line_coords[(cluster, target)]
                        target_y = ttop + target.ysize * self.cluster_margin + target.ysize * (
                                    1 - 2 * self.cluster_margin) * rel_line_coords[(target, cluster)]
                        context.move_to(cluster.pos[0], source_y)

                        context.curve_to(cluster.pos[0] + self.xseparation * 0.3, source_y,
                                         target.pos[0] - self.xseparation * 0.3, target_y, target.pos[0], target_y)

                        # context.line_to(target.pos[0], tmid)
                        context.stroke()

            context.stroke()

            context.set_line_width(self.cluster_width)
            context.set_source_rgb(0., 0., 0.)

            for t in range(self.g.num_steps):
                for cluster in self.clusters[t]:
                    cx, cy = cluster.pos
                    context.move_to(cx, cy - cluster.ysize / 2.)
                    context.line_to(cx, cy + cluster.ysize / 2.)

            context.stroke()
