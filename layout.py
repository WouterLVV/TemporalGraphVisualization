from TimeGraph import TimeGraph, TimeCluster
import cairo

class SugiyamaCluster:
    def __init__(self, tc: TimeCluster):
        self.tc = tc
        tc.sc = self

        self.rank = self.tc.id
        self.root = self
        self.align = self
        self.ys = []

        self.ysize = 1.

        self.pos = (-1., -1.)
        self.ypos = -1.

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
        self.xseparation = 50.0

        self.xmargin = 1.
        self.ymargin = 1.

        self.cluster_height_scale = 0.4

        self.bottom = 0.

        self.cluster_width = 2.
        self.line_width = 0.2
        self.line_separation = 0.1

        self.cluster_margin = 0.5-self.line_width-self.line_separation


    # ---------------- Helper Functions -------------------#

    def pred(self, c: SugiyamaCluster):
        """Returns the predecessor of this cluster (e.g. the cluster with rank-1)
        
        :param c: Cluster to find the predecessor of
        :return: SugiyamaCluster or None if no predecessor exists
        """
        if c.rank == 0:
            return None
        return self.orders[c.tc.layer][c.rank - 1]

    def largest_median_incoming(self, c: SugiyamaCluster, lower=True):
        """Returns the cluster with the largest connection to this one.

        If multiple candidates with equal connection weight exist, returns the lower median in ordering

        :param c: Cluster to find the incoming median of
        :return: The cluster and the weight of the connection
        """
        if c.insize == 0:
            return None, 0

        incomings = list(c.incoming.items())
        incomings.sort(key=lambda x: (x[1], x[0].rank), reverse=True)
        ptr = 0
        while ptr < len(incomings) and incomings[ptr][1] == incomings[0][1]:
            ptr += 1

        brother = incomings[(ptr - (1 if lower else 0)) // 2][0]
        connsize = incomings[0][1]
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
        self.orders = orders
        self.expand3()

    def barycenter_pass(self, forward, orders):
        if forward:
            r = range(1, self.num_layers)
        else:
            r = range(self.num_layers - 2, -1, -1)

        for t in r:
            order = orders[t]
            for cluster in order:

                if forward:
                    if cluster.insize > 0:
                        cluster.rank = sum([n.rank * l for n, l in cluster.incoming.items()]) / cluster.insize
                else:
                    if cluster.outsize > 0:
                        cluster.rank = sum([n.rank * l for n, l in cluster.outgoing.items()]) / cluster.outsize

            order.sort(key=lambda c: c.rank)
            for i, cluster in enumerate(order):
                cluster.rank = i

        return orders

    def place_plock(self, v):
        if v.ypos < 0.:
            v.ypos = 0.
            w = v

            while True:
                if w.rank > 0:
                    u = self.pred(w)
                    self.place_plock(u.root)

                    # # Sink stuff does not work properly
                    # if v.sink == v:
                    #     v.sink = u.sink
                    # if v.sink != u.sink:
                    #     u.sink.shift = min(u.sink.shift, v.ypos - u.ypos - self.yseparation)
                    # else:
                    v.ypos = max(v.ypos, u.root.ypos + self.yseparation + (v.ysize + u.ysize)/2.)

                w = w.align
                if w == v:
                    break

    def expand3(self, ignore_loners=False):
        inf = float('inf')
        for tc in self.clusters:
            for cluster in tc:
                cluster.align = cluster
                cluster.root = cluster
                # cluster.sink = cluster
                # cluster.shift = inf
                cluster.ysize = len(cluster.tc) * self.cluster_height_scale
                cluster.ypos = -1.

        for t in range(1, self.g.num_steps):
            r = -1

            for cluster in self.orders[t]:
                # Find cluster in previous layer this one wants to connect to and the weight of the connection
                brother, connsize = self.largest_median_incoming(cluster)

                if cluster.align == cluster and brother is not None:

                    allowed = True

                    # Check if this connection contradicts another alignment
                    # priority to the new connection is only given if the weight is higher than all crossings
                    if brother.rank <= r:
                        for i in range(brother.rank, r + 1):
                            prev_cluster = self.orders[t - 1][i]
                            if prev_cluster.align != prev_cluster.root and prev_cluster.outgoing[prev_cluster.align] > connsize:
                                allowed = False
                                break

                        # If the new connection is allowed to exist, first remove all current connections that cross it
                        if allowed:
                            for i in range(brother.rank, r + 1):
                                prev_cluster = self.orders[t - 1][i]
                                if prev_cluster.align != prev_cluster.root: # prev_cluster must not be an endpoint
                                    node_to_reset = prev_cluster.align
                                    node_to_reset.align = node_to_reset
                                    node_to_reset.root = node_to_reset
                                    prev_cluster.align = prev_cluster.root

                    if allowed:
                        brother.align = cluster
                        cluster.root = brother.root
                        cluster.align = cluster.root

                        r = brother.rank


        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            if cluster.root == cluster:
                self.place_plock(cluster)
        self.bottom = 0.
        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            cluster.ypos = cluster.root.ypos
            # if cluster.root.sink.shift < inf:
            #     cluster.ypos += cluster.root.sink.shift
            cluster.pos = (cluster.tc.layer * self.xseparation + self.xmargin,
                           cluster.ypos)  # + (cluster.root.sink.shift if cluster.root.sink.shift < inf else 0.))

        down = min([cluster.pos[1] for tc in self.clusters for cluster in tc]) - self.ymargin
        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            cluster.pos = (cluster.pos[0], cluster.pos[1] - down)

        self.bottom = max([cluster.pos[1] for tc in self.clusters for cluster in tc])
        print("done")
        for (i, order) in enumerate(self.orders):
            prev = -1
            for cluster in order:
                if cluster.pos[1] <= prev:
                    print("wtf")
                prev = cluster.pos[1]


    # ---------------- Drawing Functions ------------------- #

    def draw_graph(self, ignore_loners=False, marked_nodes=None, max_iterations=100):
        if marked_nodes is None:
            marked_nodes = set()
        self.set_locations(ignore_loners, max_pass=max_iterations)

        with cairo.SVGSurface("output/example.svg", (self.g.num_steps) * self.xseparation * self.scale + 2 * self.xmargin,
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
                        rel_line_coords[(cluster, c)] = (ctr + width/2)/cluster.insize
                        ctr += width

                    ctr = 0
                    for (c, members) in outgoings:
                        width = members
                        rel_line_coords[(cluster, c)] = (ctr + width / 2) / cluster.outsize
                        ctr += width

            for t in range(self.g.num_steps-1):
                for cluster in self.clusters[t]:

                    stop = cluster.pos[1] - cluster.ysize/2.
                    outgoings = list(cluster.outgoing.items())

                    for (target, members) in outgoings:

                        weight = members
                        context.set_line_width(self.line_width * weight)
                        if len(marked_nodes.intersection(cluster.tc.outgoing[target.tc])) > 0:
                            context.set_source_rgb(0., 0., 1.)
                        else:
                            context.set_source_rgb(1., 0., 0.)

                        ttop = target.pos[1] - target.ysize/2.

                        source_y = stop + cluster.ysize*self.cluster_margin + cluster.ysize * (1-2*self.cluster_margin) * rel_line_coords[(cluster, target)]
                        target_y = ttop + target.ysize*self.cluster_margin + target.ysize * (1-2*self.cluster_margin) * rel_line_coords[(target, cluster)]
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

