import matplotlib.pyplot as plt
from collections import deque
import io_operations

def make_node_order_random(pair_contacts):
	""" For the numerical node IDs seen in the contact data, creates a
		translation to random IDs.
	"""

	from random import shuffle # TODO: fix seed

	# collect all unique node IDs in a sorted list
	node_set = set()
	for t in pair_contacts:
		for (u, v) in pair_contacts[t]:
			node_set.add(u)
			node_set.add(v)
	node_list = sorted(list(node_set))

	# create new IDs
	order = list(range(len(node_list)))
	shuffle(order) # in place

	# returns the translation
	return { node_list[i] : order[i] for i in range(len(node_list))}

def collect_all_node_IDs(pair_contacts):
	nodes = set()
	for t in pair_contacts:
		for (u, v) in pair_contacts[t]:
			nodes.add(u)
			nodes.add(v)

	return nodes

def count_contacts_per_pair(pair_contacts):
	contact_counts = {} # {node1: {node2: #contacts, node3: ...}, node2: ...}
	for t in pair_contacts:
		for (u, v) in pair_contacts[t] + \
				list(map(lambda t: (t[1], t[0]), pair_contacts[t])):
			neighbours_of_u = contact_counts.get(u, {})
			count_of_uv = neighbours_of_u.get(v, 0)
			neighbours_of_u[v] = count_of_uv + 1
			contact_counts[u] = neighbours_of_u
	for u in contact_counts:
		print(u, contact_counts[u])

	return contact_counts

def find_highest_degree_node(contact_counts, nodes_done):
	degrees = {} # {node: #contacts}
	for u in contact_counts:
		if u not in nodes_done:
			degree_u = 0
			for v in contact_counts[u]:
				degree_u += contact_counts[u][v]
			degrees[u] = degree_u
	print(degrees)

	degree_max, node_max = -1, None
	for u in degrees:
		if degrees[u] > degree_max:
			degree_max = degrees[u]
			node_max = u
	print(degree_max, node_max)

	return node_max

def make_partial_node_order_recurring_neighbours(contact_counts, node_max, nodes_todo, nodes_done):
	order = deque([node_max])

	while nodes_todo:
		found = False
		for side in [0, -1]:
			root = order[side] # -1 is the right side
			degree_max, node = -1, None
			for u in contact_counts[root]:
				if contact_counts[root][u] > degree_max and u in nodes_todo:
					degree_max = contact_counts[root][u]
					node = u
			if node:
				if side == -1:
					order.append(node)
				else:
					order.appendleft(node)					
				nodes_todo.remove(node)
				found = True
				print(root, "->", node, "#", degree_max, "\t", order)
			else:
				print(root, "-> X")
		if not found:
			break

	nodes_done |= set(list(order))
	return order

def make_node_order_recurring_neighbours(pair_contacts):
	""" For the numerical node IDs seen in the contact data, creates a
		(greedy heuristic) translation to IDs such that nodes in contact 
		remain adjacent [17-SAC-Linhares].
	"""

	nodes_todo = collect_all_node_IDs(pair_contacts)
	contact_counts = count_contacts_per_pair(pair_contacts)
	node_max = find_highest_degree_node(contact_counts, set())
	node_list = sorted(list(nodes_todo))

	network_size = len(nodes_todo)
	network_size_done = 0
	order = []

	nodes_todo.remove(node_max)
	nodes_done = set()
	partial_order = make_partial_node_order_recurring_neighbours(contact_counts, node_max, nodes_todo, nodes_done)
	network_size_done += len(partial_order)
	order.extend(list(partial_order))
	print(partial_order)

	while network_size_done < network_size:
		node_max = find_highest_degree_node(contact_counts, nodes_done)
		nodes_todo.remove(node_max)
		partial_order = make_partial_node_order_recurring_neighbours(contact_counts, node_max, nodes_todo, nodes_done)
		network_size_done += len(partial_order)
		order.extend(list(partial_order))
		print(partial_order)

	print(order)
	# returns the translation
	return { node_list[i] : order.index(node_list[i]) for i in range(len(node_list))}

def sequence_view(pair_contacts, node_order=None):
	""" Plots a Massive Sequence View (MSV).

	Parameters
	----------
		pair_contacts : dict
		node_order : dict
	"""

	for t in pair_contacts:
		for (u, v) in pair_contacts[t]:
			coord_y1, coord_y2 = u, v
			if node_order:
				coord_y1, coord_y2 = node_order[u], node_order[v]
			# plots a line in random colour
			plt.plot((t, t), (coord_y1, coord_y2))
	plt.show()

def temporal_activity_map():
	"""
	"""

if __name__ == '__main__':
	fin_name = "tnet_sources/sociopatterns/co-presence/tij_pres_LH10.dat"
	fout_name = "Hospital"
	pair_contacts = io_operations.read_pair_contacts_from_file(fin_name, 12000, False)
	# aggregate_pair_contacts = io_operations.aggregate_time(pair_contacts, new_period=60)

	# rno = make_node_order_random(pair_contacts)
	nno = make_node_order_recurring_neighbours(pair_contacts)
	print(nno)
	sequence_view(pair_contacts, nno)

