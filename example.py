from TimeGraph import TimeGraph
from layout import SugiyamaLayout
import csv
import random

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

# with open("data/tij_pres_LyonSchool.dat", 'r') as f:
#     reader = csv.reader(f, delimiter=" ")
#     data = [(int(d[1]), int(d[2]), int((int(d[0])-34240)/20)) for d in reader if int(d[0]) < 40000]
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
#     track.add(41)
#     track.add(100)
#     print(track)
#     sg = SugiyamaLayout(g, minimum_cluster_size=2, minimum_connections_size=2)
#     sg.draw_graph(ignore_loners=False, marked_nodes=track, max_iterations=50)


with open("data/tij_pres_LyonSchool.dat", 'r') as f:
    random.seed(12345)
    reader = csv.reader(f, delimiter=" ")
    data = [(int(d[1]), int(d[2]), int((int(d[0])-34240)/20)) for d in reader if int(d[0]) < 40000]
    nodes = [d[0] for d in data]
    nodes.extend([d[1] for d in data])
    nodes = list(set(nodes))
    m = dict([(v,i) for (i,v) in enumerate(nodes)])
    data = [(m[d[0]], m[d[1]], d[2]) for d in data]
    nodenames = [str(random.randint(1, 4)) for _ in nodes]
    num_steps = max([d[2] for d in data])+1
    g = TimeGraph(data, nodenames, num_steps)
    sg = SugiyamaLayout(g, minimum_cluster_size=1, minimum_connections_size=1)
    sg.draw_graph(ignore_loners=False, max_iterations=50, colormap={"1": (1., 0.5, 0.5, 1.), "2": (0.5, 1., 0.5, 1.), "3": (0.5, 0.5, 1., 1.), "4": (0.5, 0.5, 0.5, 1.) })

#
# with open("data/primaryschool.csv", 'r') as f:
#     reader = csv.reader(f, delimiter="\t")
#     data = [(int(d[1]), int(d[2]), (int(d[0])-31220)//20, d[3], d[4]) for d in reader if int(d[0]) < 32000]
#     nodes = [(d[0], d[3]) for d in data]
#     nodes.extend([(d[1], d[4]) for d in data])
#     nodes = list(set(nodes))
#     m = dict([(v[0],i) for (i,v) in enumerate(nodes)])
#     data = [(m[d[0]], m[d[1]], d[2]) for d in data]
#     nodenames = [n for _, n in nodes]
#     num_steps = max([d[2] for d in data])+1
#     g = TimeGraph(data, nodenames, num_steps)
# #    pprint(g.clustered_graph())
# #     track = set()
# #     track.add(2)
# #     print(track)
#     sg = SugiyamaLayout(g, minimum_cluster_size=1, minimum_connections_size=1)
#     sg.draw_graph(ignore_loners=False, max_iterations=50, colormap={"4A": (0., 1., 0., 1.), "5A": (1., 0.5, 0., 1.), "Teachers": (0., 0., 1., 1.), "3B": (0., 0.5, 0., 1.)})

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