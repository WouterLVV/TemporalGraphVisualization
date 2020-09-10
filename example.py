from TimeGraph import TimeGraph
from layout import SugiyamaLayout
import csv

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

with open("data/tij_pres_LyonSchool.dat", 'r') as f:
    reader = csv.reader(f, delimiter=" ")
    data = [(int(d[1]), int(d[2]), int((int(d[0])-34240)/20)) for d in reader if int(d[0]) < 36000]
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
    track.add(41)
    track.add(100)
    print(track)
    sg = SugiyamaLayout(g)
    sg.draw_graph(False, track, 50)

# with open("data/primaryschool.csv", 'r') as f:
#     reader = csv.reader(f, delimiter="\t")
#     data = [(int(d[1]), int(d[2]), int((int(d[0])-31220)/20)) for d in reader if int(d[0]) < 32000]
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
#     track.add(2)
#     print(track)
#     g.draw_graph(False, track)

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