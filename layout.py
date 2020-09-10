from TimeGraph import TimeGraph, TimeCluster
import cairo

class SugiyamaCluster:
    def __init__(self, tc: TimeCluster):
        self.tc = tc

        self.rank = self.tc.id
        self.root = self
        self.align = self
        self.ys = []

        self.pos = (-1., -1.)
        self.ypos = -1.


class SugiyamaLayout:
    def __init__(self, g: TimeGraph):
        self.g = g
        self.orders = [self.g.clusters[t].copy() for t in range(self.g.num_steps)]
        self.clusters = [[SugiyamaCluster(self.g.clusters[t][i]) for i in range(len(self.g.clusters[t]))] for t in
                         range(self.g.num_steps)]

        self.num_layers = self.g.num_steps

        self.scale = 2.
        self.yseparation = 5.0
        self.xseparation = 50.0

        self.xmargin = 1.
        self.ymargin = 1.

        self.bottom = 0.


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
        incomings = list(c.tc.incoming.items())
        incomings.sort(key=lambda x: (len(x[1]), self.clusters[c.tc.layer - 1][x[0]].rank), reverse=True)
        ptr = 0
        while ptr < len(incomings) and len(incomings[ptr][1]) == len(incomings[0][1]):
            ptr += 1

        brother = self.clusters[c.tc.layer - 1][incomings[(ptr - (1 if lower else 0)) // 2][0]]
        connsize = len(incomings[0][1])
        return brother, connsize


    # ---------------- Location Functions -------------------#

    def set_locations(self, ignore_loners=False, max_pass=100):

        # First layer is initialized on their id
        for i, cluster in enumerate(self.clusters[0]):
            if ignore_loners and len(cluster.tc) == 1:
                continue
            cluster.pos = (self.xmargin, i)
            cluster.rank = i


        forward = True
        pass_ctr = 1
        orders_old = self.barycenter_pass(forward, ignore_loners)
        orders_new = None
        forward = False

        # Keep doing passes until the maximum number has been reached or the order does no longer change
        # Always end with a forward pass
        while pass_ctr < max_pass or forward:
            orders_new = self.barycenter_pass(forward, ignore_loners)
            if forward:
                unchanged = True
                for t in range(self.num_layers - 1):
                    unchanged &= orders_old[t] == orders_new[t]
                    if not unchanged:
                        break
                if unchanged:
                    break
                orders_old = orders_new

            pass_ctr += 1
            forward = not forward
            print(pass_ctr)

        self.expand3(orders_new)

    def barycenter_pass(self, forward, ignore_loners=False):
        if forward:
            r = range(1, self.num_layers)
            s = 0
            prev = 1
        else:
            r = range(self.num_layers - 2, -1, -1)
            s = self.num_layers - 1
            prev = -1
        orders = []
        order = self.clusters[s].copy()
        order.sort(key=lambda c: c.rank)
        orders.append(order)
        for t in r:
            order = []
            for cluster in self.clusters[t]:
                if ignore_loners and len(cluster.tc) == 1:
                    continue
                cluster.rank = sum([self.clusters[t - prev][self.g.nodes[n].clusters[t - prev]].rank for n in
                                    cluster.tc.members]) / len(cluster.tc)
                order.append(cluster)
            order.sort(key=lambda c: c.rank)
            for i, cluster in enumerate(order):
                cluster.rank = i
            orders.append(order)
        return orders

    def place_plock(self, v):
        if v.ypos < 0.:
            v.ypos = 0.
            w = v

            while True:
                if w.rank > 0:
                    u = self.pred(w).root
                    self.place_plock(u)

                    # # Sink stuff does not work properly
                    # if v.sink == v:
                    #     v.sink = u.sink
                    # if v.sink != u.sink:
                    #     u.sink.shift = min(u.sink.shift, v.ypos - u.ypos - self.yseparation)
                    # else:
                    v.ypos = max(v.ypos, u.ypos + self.yseparation)

                w = w.align
                if w == v:
                    break

    def expand3(self, orders, ignore_loners=False):
        inf = float('inf')
        for tc in self.clusters:
            for cluster in tc:
                cluster.align = cluster
                cluster.root = cluster
                cluster.sink = cluster
                cluster.shift = inf
                cluster.ysize = len(cluster.tc)
                cluster.ypos = -1.

        for t in range(1, self.g.num_steps):
            r = -1

            for cluster in orders[t]:
                # Find cluster in previous layer this one wants to connect to and the weight of the connection
                brother, connsize = self.largest_median_incoming(cluster)

                if cluster.align == cluster:

                    allowed = True

                    # Check if this connection contradicts another alignment
                    # priority to the new connection is only given if the weight is higher than all crossings
                    if brother.rank <= r:
                        for i in range(brother.rank, r + 1):
                            prev_cluster = orders[t - 1][i]
                            if prev_cluster.align != prev_cluster.root and len(
                                    prev_cluster.outgoing[prev_cluster.align.id]) > connsize:
                                allowed = False
                                break

                        # If the new connection is allowed to exist, first remove all current connections that cross it
                        if allowed:
                            for i in range(brother.rank, r + 1):
                                prev_cluster = orders[t - 1][i]
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
            if cluster.layer == 4 and cluster.id == 14:
                print("poepie")
            cluster.ypos = cluster.root.ypos
            # if cluster.root.sink.shift < inf:
            #     cluster.ypos += cluster.root.sink.shift
            cluster.pos = (cluster.layer * self.xseparation + self.xmargin,
                           cluster.ypos)  # + (cluster.root.sink.shift if cluster.root.sink.shift < inf else 0.))

        down = min([cluster.pos[1] for tc in self.clusters for cluster in tc]) - self.ymargin
        for cluster in [cluster for tc in self.clusters for cluster in tc]:
            cluster.pos = (cluster.pos[0], cluster.pos[1] - down)

        self.bottom = max([cluster.pos[1] for tc in self.clusters for cluster in tc])
        print("done")
        for (i, order) in enumerate(orders):
            prev = -1
            for cluster in order:
                if cluster.pos[1] <= prev:
                    print("wtf")
                prev = cluster.pos[1]


    # ---------------- Drawing Functions -------------------#

    def draw_graph(self, ignore_loners=False, marked_nodes=None, max_iterations=100):
        if marked_nodes is None:
            marked_nodes = set()
        self.set_locations(ignore_loners, max_pass=max_iterations)

        with cairo.SVGSurface("output/example.svg", (self.g.num_steps) * self.xseparation * self.scale + 2 * self.xmargin,
                              self.bottom * self.scale + 2 * self.ymargin) as surface:
            context = cairo.Context(surface)
            context.scale(self.scale, self.scale)

            context.set_source_rgb(1., 0., 0.)
            context.set_line_width(0.1)
            for t in range(self.g.num_steps - 1):
                for cluster in self.clusters[t]:
                    if ignore_loners and len(cluster) <= 1:
                        continue

                    smid = cluster.pos[1]

                    for (target_id, members) in cluster.outgoing.items():
                        target = self.clusters[t + 1][target_id]
                        if ignore_loners and len(target) == 1:
                            continue
                        weight = len(members)
                        context.set_line_width(0.1 * weight)
                        if len(marked_nodes.intersection(members)) > 0:
                            context.set_source_rgb(0., 0., 1.)
                        else:
                            context.set_source_rgb(1., 0., 0.)

                        tmid = target.pos[1]
                        context.move_to(cluster.pos[0], smid)

                        context.curve_to(cluster.pos[0] + self.xseparation * 0.3, smid,
                                         target.pos[0] - self.xseparation * 0.3, tmid, target.pos[0], tmid)

                        # context.line_to(target.pos[0], tmid)
                        context.stroke()

            context.stroke()

            context.set_line_width(0.3)

            for t in range(self.g.num_steps):
                for cluster in self.clusters[t]:
                    cx, cy = cluster.pos
                    context.move_to(cx, cy - cluster.ysize / 2.)
                    context.line_to(cx, cy + cluster.ysize / 2.)

            context.stroke()

