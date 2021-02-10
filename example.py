# To be replaced with small examples
from tgv.timegraph import TimeGraph
from tgv.layout import SizedConnectionLayout

"""
This module contains tools in order to display communities that evolve over time,
It is designed for proximity dataset, but can be used for any sort of data that evolves over time.

Fist it is up to the user to make sure that the data is divided in steps. Each step contains information about which
two people were in contact during that timestep. The most simple input is a list where the entries consist of a timestep
and two node identifiers.

For example:
"""
contactlist = [
    ('a', 'b', 0),
    ('b', 'c', 0),
    ('b', 'c', 1)
]
"""
The node identifiers can be ints, strings or arbitrary objects.

In this example, person a and person b are in contact and b and c are in contact in timestep 0. Some amount of time 
passes and in the next step b and c are in contact, but b no longer is in contact with a. 
"""
tg = TimeGraph(contactlist)

"""
The TimeGraph will take the contact information and make clusters of people who were near eachother in a given timestep. 
"""
print("1: ", [(cluster.layer, list(map(str, cluster.members))) for cluster in tg.clusters])

"""And lets see what it looks like:"""
SizedConnectionLayout(tg).draw_graph("example1.svg", scale=10.)

"""
In step 0, A was near B and B was near C so A is considered to be in the same cluster as C even though there was no 
direct contact. Similarly, even though we did not specify and info about A in step 1, there is still a cluster of 1.

Because in these datasets it is possible that even though there is no information, there is still data.
For example, what if there is a person d that did not have any contacts whatsoever, or a timestep where the was no contact?
You can specify the nodes and number of steps as follows.
"""
nodes = ['a', 'b', 'c', 'd']
num_steps = 3

tg = TimeGraph(contactlist, nodes=nodes, num_steps=num_steps)

print("\n2: ", [(cluster.layer, list(map(str, cluster.members))) for cluster in tg.clusters])
SizedConnectionLayout(tg).draw_graph("example2.svg", scale=10.)

"""
Additionally, in larger datasets you may not want to have to pay attention to smaller clusters or the smaller movements
between clusters over time. For this there is the minimum cluster size and minimum connection size. Clusters below the
threshold or which don't have any connections are removed from the graph.
"""

tg = TimeGraph(contactlist, nodes=nodes, num_steps=num_steps, minimum_cluster_size=2)

print("\n3: ", [(cluster.layer, list(map(str, cluster.members))) for cluster in tg.clusters])
SizedConnectionLayout(tg).draw_graph("example3.svg", scale=10.)

"""
Instead of only an identifier, we can give each node an id and metadata. The timegraph does not use the metadata, but
the drawing can separate out colors if those are specified.
Colors are specified in the RGBA format, with values between 0 and 1.
"""

nodes_with_metadata = {'a': "group1", 'b': "group1", 'c': "group2", 'd': "group2"}
colors = {"group1": (0., 0.5, 1., 1.), "group2": (0.75, 0.4, 0.2, 1.)}

tg = TimeGraph(contactlist, nodes=nodes_with_metadata)
SizedConnectionLayout(tg).draw_graph("example4.svg", scale=10., colormap=colors)