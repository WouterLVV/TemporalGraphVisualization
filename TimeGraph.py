from collections import deque
from pprint import pprint
import cairo
import csv

class TimeNode:
    def __init__(self, conns : set, max_step : int, id = -1, name = None):
        self.id = id
        self.name = name

        self.nbs = [set() for _ in range(max_step)]
        self.clusters = [-1]*num_steps

        if conns is not None:
            for (n, t) in conns:
                self.nbs[t].add(n)

    def add_connection(self, n, t):
        self.nbs[t].add(n)



class TimeGraph:
    def __init__(self, conns, num_nodes : int, num_steps : int, name_map=None):
        self.name_map = name_map

        self.num_steps = num_steps
        self.num_nodes = num_nodes

        self.nodes = [TimeNode(None, num_steps, id) for id in range(num_nodes)]

        self.clusters = []

        self.scale = 2.
        self.yseparation = 1.0
        self.xseparation = 10.0

        self.xmargin = 1.
        self.ymargin = 1.

        self.bottom = 0.

        for (f,b,t) in conns:
            self.nodes[f].add_connection(b, t)
            self.nodes[b].add_connection(f, t)

        self.build()

    def components_t(self, t : int):
        clusts = []
        seen = set()
        ctr = 0
        for node in self.nodes:
            if node.id in seen:
                continue

            seen.add(node.id)
            clust = TimeCluster(self, t, ctr)
            q = deque()
            q.append(node.id)
            while len(q) > 0:
                n_id = q.pop()

                clust.add(n_id)
                self.nodes[n_id].clusters[t] = ctr
                for nb in self.nodes[n_id].nbs[t]:
                    if nb not in seen:
                        q.append(nb)
                        seen.add(nb)

            clusts.append(clust)
            ctr += 1
        return clusts

    def components(self):
        return [self.components_t(t) for t in range(self.num_steps)]

    def build(self):
        self.clusters = self.components()
        self.connect_clusters()


    def order(self):

        order = [[(i,i) for i in range(len(self.clusters[0]))]]
        lookup = [[i for i in range(len(self.clusters[0]))]]
        for i in range(1, self.num_steps):
            o = [(sum([lookup[i-1][self.nodes[n].clusters[i-1]] for n in c.members])/len(c), c.id) for c in self.clusters[i]]
            o.sort()
            l = [0]*len(o)
            for k, c in enumerate(o):
                l[c[1]] = k

            order.append(o)
            lookup.append(l)
        return order, lookup

    def connect_clusters(self):
        for n_id in range(len(self.nodes)):
            for t in range(self.num_steps - 1):

                connection = (self.nodes[n_id].clusters[t], self.nodes[n_id].clusters[t+1])
                self.clusters[t][connection[0]].add_connection(connection[1], t+1, n_id)
                self.clusters[t+1][connection[1]].add_connection(connection[0], t, n_id)


    # def set_locations(self, ignore_loners=False):
    #     order, lookup = self.order()
    #
    #     for t in range(self.num_steps):
    #         x = float(t) * self.xseparation + self.xmargin
    #         y = self.ymargin
    #         for (_, c) in order[t]:
    #             if ignore_loners and len(self.clusters[t][c]) == 1:
    #                 continue
    #             cluster = self.clusters[t][c]
    #             y += self.yseparation
    #             cluster.ysize = len(cluster)
    #             cluster.pos = (x, y)
    #             y += cluster.ysize

    def set_locations(self, ignore_loners=False, max_pass=498):

        # y = self.ymargin
        # for cluster in self.clusters[0]:
        #     if ignore_loners and len(cluster) == 1:
        #         continue
        #     cluster.pos = (self.xmargin, y)
        #     cluster.ysize = float(len(cluster))
        #     y += cluster.ysize + self.yseparation

        for i,cluster in enumerate(self.clusters[0]):
            if ignore_loners and len(cluster) == 1:
                continue
            cluster.pos = (self.xmargin, i)



        forward = True
        pass_ctr = 1
        # orders_old = self.median_pass(forward, ignore_loners)
        orders_old = self.barycenter_pass(forward, ignore_loners)
        while pass_ctr < max_pass:
            forward = not forward
            # orders_new = self.median_pass(forward, ignore_loners)
            orders_new = self.barycenter_pass(forward, ignore_loners)
            if not forward:
                unchanged = True
                for t in range(self.num_steps):
                    unchanged &= orders_old[t] == orders_new[t]
                    if not unchanged:
                        break
                if unchanged:
                    break
                orders_old = orders_new

            pass_ctr += 1
            print(pass_ctr)

        self.expand(ignore_loners)
        # self.forward_pass(ignore_loners)
        # self.backward_pass(ignore_loners)
        # # self.forward_pass(ignore_loners)


    def barycenter_pass(self, forward=True, ignore_loners=False):
        if forward:
            r = range(1, self.num_steps)
            prev = 1
        else:
            r = range(self.num_steps - 2, -1, -1)
            prev = -1
        orders = []
        for t in r:
            order = []
            for cluster in self.clusters[t]:
                if ignore_loners and len(cluster) == 1:
                    continue
                avg = sum([self.clusters[t-prev][self.nodes[n].clusters[t-prev]].pos[1] for n in cluster.members])/len(cluster)
                order.append((avg, cluster.id))
            order.sort()
            for i, (_, c_id) in enumerate(order):
                cluster = self.clusters[t][c_id]
                cluster.pos = (cluster.pos[0], i)
            orders.append(order)
        return orders

    def median_pass(self, forward=True, ignore_loners=False):
        if forward:
            r = range(1, self.num_steps)
            prev = 1
        else:
            r = range(self.num_steps - 2, -1, -1)
            prev = -1

        orders = []
        for t in r:
            order = []
            for cluster in self.clusters[t]:
                if ignore_loners and len(cluster) == 1:
                    continue
                prevs =  [self.clusters[t - prev][self.nodes[n].clusters[t - prev]].pos[1] for n in cluster.members]
                prevs.sort()
                med = prevs[len(cluster.members)//2]
                order.append((med, cluster.id))
            order.sort()
            for i, (_, c_id) in enumerate(order):
                cluster = self.clusters[t][c_id]
                cluster.pos = (cluster.pos[0], i)
            orders.append(order)
        return orders



    def expand(self, ignore_loners=False):


        for t in range(0, self.num_steps):
            c_order = [(c.pos[1], c.id) for c in self.clusters[t]]
            c_order.sort()

            y = self.ymargin
            for i, (_, c_id) in enumerate(c_order):
                cluster = self.clusters[t][c_id]
                if ignore_loners and len(cluster) == 1:
                    continue
                cluster.ysize = float(len(cluster))*self.yseparation
                cluster.pos = (self.xseparation*t + self.xmargin, y)
                y += cluster.ysize + self.yseparation

            if y > self.bottom:
                self.bottom = y

    def forward_pass(self, ignore_loners=False):
        for t in range(1,self.num_steps):
            order = []
            for cluster in self.clusters[t]:
                if ignore_loners and len(cluster) == 1:
                    continue
                cluster.pos = (self.xseparation*t + self.xmargin, sum([self.clusters[t-1][self.nodes[n].clusters[t-1]].pos[1] for n in cluster.members])/len(cluster))
                cluster.ysize = float(len(cluster))
                order.append((cluster.pos[1], cluster.id))
            order.sort()
            prev_bottom = 0.
            prev_top_margin = 0.
            for i, (_, c_id) in enumerate(order):
                cluster = self.clusters[t][c_id]
                wantedtop = cluster.pos[1]
                if wantedtop < self.yseparation + prev_bottom:
                    diff = self.yseparation + prev_bottom - wantedtop
                    if (diff < prev_top_margin):
                        prev_cluster = self.clusters[t][order[i-1][1]]
                        prev_cluster.pos = (prev_cluster.pos[0], prev_cluster.pos[1] - diff)
                    else:
                        cluster.pos = (cluster.pos[0], prev_bottom + self.yseparation)
                    prev_top_margin = 0.
                else:
                    prev_top_margin = wantedtop - self.yseparation - prev_bottom
                prev_bottom = cluster.pos[1] + cluster.ysize
            if prev_bottom > self.bottom:
                self.bottom = prev_bottom

    def backward_pass(self, ignore_loners=False):
        for t in range(self.num_steps-2, -1, -1):
            order = []
            for cluster in self.clusters[t]:
                if ignore_loners and len(cluster) == 1:
                    continue
                cluster.pos = (self.xseparation * t + self.xmargin, sum([self.clusters[t + 1][self.nodes[n].clusters[t + 1]].pos[1] for n in cluster.members]) / len(cluster))
                order.append((cluster.pos[1], cluster.id))
            order.sort()
            prev_bottom = 0.
            prev_top_margin = 0.
            for i, (_, c_id) in enumerate(order):
                cluster = self.clusters[t][c_id]
                wantedtop = cluster.pos[1]
                if wantedtop < self.yseparation + prev_bottom:
                    if (self.yseparation + prev_bottom - wantedtop < prev_top_margin):
                        prev_cluster = self.clusters[t][order[i - 1][1]]
                        prev_cluster.pos = (
                        prev_cluster.pos[0], prev_cluster.pos[1] - self.yseparation + prev_bottom - wantedtop)
                    else:
                        cluster.pos = (cluster.pos[0], prev_bottom + self.yseparation)
                    prev_top_margin = 0.
                else:
                    prev_top_margin = wantedtop - self.yseparation - prev_bottom
                prev_bottom = cluster.pos[1] + cluster.ysize
            if prev_bottom > self.bottom:
                self.bottom = prev_bottom

    def draw_graph(self, ignore_loners=False, marked_nodes=None):
        if marked_nodes is None:
            marked_nodes = set()
        self.set_locations(ignore_loners)

        with cairo.SVGSurface("example.svg", (self.num_steps) * self.xseparation * self.scale + 2 * self.xmargin, self.bottom * self.scale + 2 * self.ymargin) as surface:
            context = cairo.Context(surface)
            context.scale(self.scale, self.scale)
            context.set_line_width(0.3)

            for t in range(num_steps):
                for cluster in self.clusters[t]:
                    cx, cy = cluster.pos
                    context.move_to(cx, cy)
                    context.line_to(cx, cy + cluster.ysize)

            context.stroke()

            context.set_source_rgb(1.,0.,0.)
            context.set_line_width(0.1)
            for t in range(self.num_steps-1):
                for cluster in self.clusters[t]:
                    if ignore_loners and len(cluster) <= 1:
                        continue

                    smid = cluster.pos[1] + cluster.ysize/2

                    for (target_id, members) in cluster.outgoing.items():
                        target = self.clusters[t+1][target_id]
                        if ignore_loners and len(target) == 1:
                            continue
                        weight = len(members)
                        context.set_line_width(0.1*weight)
                        if len(marked_nodes.intersection(members)) > 0:
                            context.set_source_rgb(0., 0., 1.)
                        else:
                            context.set_source_rgb(1., 0., 0.)

                        tmid = target.pos[1] + target.ysize/2
                        context.move_to(cluster.pos[0], smid)

                        context.curve_to(cluster.pos[0] + self.xseparation*0.3, smid, target.pos[0] - self.xseparation*0.3, tmid, target.pos[0], tmid)

                        #context.line_to(target.pos[0], tmid)
                        context.stroke()

            context.stroke()




class TimeCluster:
    def __init__(self, parent : TimeGraph, layer: int, id : int):
        self.id = id

        self.incoming = dict()
        self.outgoing = dict()
        self.members = set()

        self.layer = layer
        self.ysize = 0.

        self.pos = (-1.,-1.)

        self.parent = parent

    def add(self, node_id : int):
        self.members.add(node_id)

    def add_connection(self, target : int, t : int, members):
        if type(members) == int:

            if members not in self.members:
                raise Exception("Connection contains members not in this cluster")

            if t == self.layer+1:
                if target not in self.outgoing.keys():
                    self.outgoing[target] = set()
                self.outgoing[target].add(members)
            elif t == self.layer-1:
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




# data = [
#     (0, 1, 0),
#     (0, 2, 0),
#     (1, 4, 0),
#     (3, 5, 0),
#     (5, 6, 0),
#     (0, 2, 1),
#     (1, 4, 1),
#     (4, 5, 1),
#     (2, 3, 1),
#     (0, 1, 2),
#     (2, 3, 2),
#     (4, 5, 2),
# ]
# num_nodes = 7
# num_steps = 3
# g = TimeGraph(data, num_nodes, num_steps)
# pprint(g.clustered_graph())
# g.draw_graph()

# with open("High-School_data_2013.csv", 'r') as f:
#     reader = csv.reader(f, delimiter=" ")
#     data = [(int(d[1]), int(d[2]), int((int(d[0])-1385982020)/20)) for d in reader if int(d[0]) < 1385989100]
#     nodes = [d[0] for d in data]
#     nodes.extend([d[1] for d in data])
#     nodes = list(set(nodes))
#     m = dict([(v,i) for (i,v) in enumerate(nodes)])
#     data = [(m[d[0]], m[d[1]], d[2]) for d in data]
#     num_nodes = len(nodes)
#     num_steps = max([d[2] for d in data])+1
#     g = TimeGraph(data, num_nodes, num_steps)
# #    pprint(g.clustered_graph())
#     track = set()
#     track.add(0)
#     print(track)
#     g.draw_graph(False, track)


with open("primaryschool.csv", 'r') as f:
    reader = csv.reader(f, delimiter="\t")
    data = [(int(d[1]), int(d[2]), int((int(d[0])-31220)/20)) for d in reader if int(d[0]) < 32000]
    nodes = [d[0] for d in data]
    nodes.extend([d[1] for d in data])
    nodes = list(set(nodes))
    m = dict([(v,i) for (i,v) in enumerate(nodes)])
    data = [(m[d[0]], m[d[1]], d[2]) for d in data]
    num_nodes = len(nodes)
    num_steps = max([d[2] for d in data])+1
    g = TimeGraph(data, num_nodes, num_steps)
#    pprint(g.clustered_graph())
    track = set()
    track.add(0)
    print(track)
    g.draw_graph(False, track)

# with open("detailed_list_of_contacts_Hospital.tsv", 'r') as f:
#     reader = csv.reader(f, delimiter="\t")
#     data = [(int(d[1]), int(d[2]), int((int(d[0]))/20)) for d in reader if int(d[0]) < 73000]
#     nodes = [d[0] for d in data]
#     nodes.extend([d[1] for d in data])
#     nodes = list(set(nodes))
#     m = dict([(v,i) for (i,v) in enumerate(nodes)])
#     data = [(m[d[0]], m[d[1]], d[2]) for d in data]
#     num_nodes = len(nodes)
#     num_steps = max([d[2] for d in data])+1
#     g = TimeGraph(data, num_nodes, num_steps)
# #    pprint(g.clustered_graph())
#     track = set()
#     track.add(0)
#     print(track)
#     g.draw_graph(False, track)