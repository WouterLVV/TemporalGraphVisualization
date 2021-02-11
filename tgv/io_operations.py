def read_pair_contacts_from_file(fin_name, separator=' ', start_timestamp=-1, end_timestamp=-1, timestamp_first=True, verbose=False):
    """ Reads a temporal networks from a text file.

    Parameters
    ----------
        fin_name : str
            input file with whitespace-separated triplets per line:
            timestamp node1 node2 (assumed to appear chronologically)
        start_timestamp, end_timestamp : int
            interval of timestamps to read from file (inclusive)
            either neither or both should be set
        verbose : bool
            if True then also print result (default False)
    Returns
    -------
        pair_contacts : dict
            key timestamp, value list of pair contacts
    """

    pair_contacts = {}  # {timestamp : list of tuples (pair in contact)}

    with open(fin_name, 'r') as fin:
        for l in fin:
            if l[0] == '#':
                continue
            if timestamp_first:
                t, u, v = l.split(separator)
            else:
                u, v, t = l.split(separator)

            t = int(t)

            if (start_timestamp < 0 or t >= start_timestamp) and (end_timestamp < 0 or t <= end_timestamp):
                if t not in pair_contacts.keys():
                    pair_contacts[t] = []
                lc = pair_contacts[t]
                u = u.strip()
                v = v.strip()

                u, v = int(u), int(v)  # not technically needed but ints work slightly faster than strings
                if u > v:
                    u, v = v, u
                lc.append((u, v))  # ensures contacts are saved in a unique form
            elif t > end_timestamp and end_timestamp >= 0:
                break

    if verbose:
        print("*** Pair contacts in", fin_name)
        tt = sorted(pair_contacts.keys())
        for t in tt:
            print("\t", t, pair_contacts[t])

    return pair_contacts


def add_missing_timestamps(pair_contacts, grain_t, start_timestamp=-1, end_timestamp=-1, verbose=False):
    # Missing timestamps are timestamps in which there were no connections and thus are not in the data
    timestamps = sorted(list(pair_contacts.keys()))

    # Extend range if timestamps are missing at the beginning or end
    if start_timestamp >= 0 and start_timestamp < timestamps[0]:
        timestamps = [start_timestamp] + timestamps
    if end_timestamp >= 0 and end_timestamp > timestamps[-1]:
        timestamps = timestamps + [end_timestamp]

    for i in range(1, len(timestamps)):
        this_t = timestamps[i]
        prev_t = timestamps[i-1]
        if this_t > prev_t + grain_t:
            for missing_t in range(prev_t+grain_t, this_t, grain_t):
                pair_contacts[missing_t] = []

    if verbose:
        print("*** Pair contacts after filling in missing")
        tt = sorted(pair_contacts.keys())
        for t in tt:
            print("\t", t, pair_contacts[t])

    return pair_contacts


def read_node_metadata_from_file(fin_name, verbose=False):
    node_metadata = {} # {node id: metadata/category as string}
    categories = set()

    with open(fin_name, 'r') as fin:
        for l in fin:
            node, metadata = l.split()
            node = int(node)
            node_metadata[node] = metadata
            categories.add(metadata)

    categories = list(categories)
    if verbose:
        print("*** Categories of nodes in metadata:\n\t", ' '.join(sorted(categories)))

    return node_metadata, sorted(categories)


def initialise_aggregation(pair_contacts, old_period, new_period):
    old_timestamps = sorted(list(pair_contacts.keys()))
    aggregate_pair_contacts = {}

    # old: a contact was active in (t–old_period, t]
    # new: a contact  is active in (t–new_period, t]

    i = 0  # index in old timestamps
    new_timestamp = old_timestamps[0] - old_period + new_period

    while i < len(old_timestamps):
        while i < len(old_timestamps) and old_timestamps[i] <= new_timestamp:
            contacts = aggregate_pair_contacts.get(new_timestamp, [])
            contacts.extend(pair_contacts[old_timestamps[i]])
            aggregate_pair_contacts[new_timestamp] = contacts
            i += 1
        new_timestamp += new_period

    return aggregate_pair_contacts

def strongly_aggregate_time(pair_contacts, old_period, new_period, strength=0.5):
    aggregate_pair_contacts = initialise_aggregation(pair_contacts, old_period, new_period)

    # removes all contacts from an aggregated timestamp which occur too infrequently;
    # the output contains only unique contact pairs in the new period 
    from collections import Counter
    min_count_wanted = strength * (new_period / old_period)

    for t in aggregate_pair_contacts:
        new_pair_contacts = []

        cntr = Counter(aggregate_pair_contacts[t])
        for contact, count in cntr.items():
            if count >= min_count_wanted:
                new_pair_contacts.append(contact)
        aggregate_pair_contacts[t] = new_pair_contacts

    return aggregate_pair_contacts

def normalise_list_pair_contacts(pair_contacts, node_metadata=None, time_label='s'):

    # collect the original node ids and timestamps
    timestamps = sorted(list(pair_contacts.keys()))

    nodeids = set()
    for t in pair_contacts:
        nodeids |= set([node for contact in pair_contacts[t] for node in contact])
    nodeids = sorted(list(nodeids))

    # timestamp translators: from original t to a 0-index, and back from 0-index to a pretty-printed t
    timestamp_translator = dict([(old, new) for (new, old) in enumerate(timestamps)])
    timestamp_reverse_translator = dict(enumerate(timestamps))

    from time import gmtime, strftime
    if len(str(timestamps[0])) == 10: # timestamps are Unix seconds? convert to string in my local time zone
        for i in timestamp_reverse_translator:
            t = timestamp_reverse_translator[i]
            timestamp_reverse_translator[i] = strftime("%H:%M", gmtime(t))
    elif time_label == 's':        # timestamps are seconds? convert to hour:minute
        for i in timestamp_reverse_translator:
            t = int(timestamp_reverse_translator[i])
            timestamp_reverse_translator[i] = str(int(t/3600)%24) + ":" + str(int((t%3600)/60)).zfill(2)
    elif time_label == 'd':        # timestamps are days? convert to day number
        for i in timestamp_reverse_translator:
            t = int(timestamp_reverse_translator[i])
            timestamp_reverse_translator[i] = str(int(t/3600/24) + 1)

    # node id translator from original node id to a 0-index, and back
    nodeid_translator = dict([(old, new) for (new, old) in enumerate(nodeids)])
    nodeid_reverse_translator = dict(enumerate(nodeids))

    # node metadata as a list following the new node ids
    node_metadata_list = []
    if node_metadata != None:
        for i in range(len(nodeids)):
            node_metadata_list.append(node_metadata[nodeid_reverse_translator[i]])

    # the normalised pair contacts
    normalised_list_pair_contacts = []
    for t in pair_contacts:
        for (u, v) in pair_contacts[t]:
            normalised_list_pair_contacts.append((nodeid_translator[u], nodeid_translator[v], timestamp_translator[t]))

    # collect results
    return normalised_list_pair_contacts, len(timestamps), len(nodeids), \
           timestamp_reverse_translator, nodeid_translator, node_metadata_list

def node_ids_from_pair_contacts(pair_contacts):
    nodeids = set()
    for t in pair_contacts:
        nodeids |= set([node for contact in pair_contacts[t] for node in contact])
    return nodeids

def normalise_timestamps(pair_contacts, time_label='s'):

    # collect the original node ids and timestamps
    timestamps = sorted(list(pair_contacts.keys()))

    # timestamp translators: from original t to a 0-index, and back from 0-index to a pretty-printed t
    timestamp_translator = dict([(old, new) for (new, old) in enumerate(timestamps)])
    timestamp_reverse_translator = dict(enumerate(timestamps))

    from time import gmtime, strftime
    if len(str(timestamps[0])) == 10:  # timestamps are Unix seconds? convert to string in my local time zone
        for i in timestamp_reverse_translator:
            t = timestamp_reverse_translator[i]
            timestamp_reverse_translator[i] = strftime("%H:%M", gmtime(t))
    elif time_label == 's':  # timestamps are seconds? convert to hour:minute
        for i in timestamp_reverse_translator:
            t = int(timestamp_reverse_translator[i])
            timestamp_reverse_translator[i] = str(int(t / 3600) % 24) + ":" + str(int((t % 3600) / 60)).zfill(2)
    elif time_label == 'd':  # timestamps are days? convert to day number
        for i in timestamp_reverse_translator:
            t = int(timestamp_reverse_translator[i])
            timestamp_reverse_translator[i] = str(int(t / 3600 / 24) + 1)

    # the normalised pair contacts
    time_normalised_pair_contacts = {timestamp_translator[t]: l for t, l in pair_contacts.items()}

    return time_normalised_pair_contacts, timestamp_reverse_translator

def null_model(aggregate_pair_contacts):
    """ Null model: degree-preserving rewiring of the unweighted contact graph of each timestamp.

    Maintains key characteristics of the data set:
      #contacts per node and per timestamp
      (so also amount of activity per timestamp, of node)
    Breaks all temporal social structure.
    """

    import igraph
    null_pair_contacts = {} # { timestamp: [pair contacts] }

    for t in aggregate_pair_contacts:

        # create the contact graph for this timestamp
        contact_list = aggregate_pair_contacts[t]
        edge_list = list(map(lambda c: (str(c[0]), str(c[1])), contact_list)) # strings needed as 'name' attributes
        vertex_set = set()
        for e in edge_list:
            vertex_set.add(e[0])
            vertex_set.add(e[1])

        if len(contact_list) < 2 or len(vertex_set) < 4:
            null_pair_contacts[t] = aggregate_pair_contacts[t]

        # try rewiring randomly, in place
        else:
            G = igraph.Graph()
            G.add_vertices(list(vertex_set))
            G.add_edges(edge_list)
            rewired = False

            try:
                G.rewire()
                rewired = True
            except:
                pass

            # save the rewired edges
            if rewired:
                rewired_edge_list = []
                for e in G.es:
                    u, v = e.tuple # these are 0-indices though, not the original node id, now in the 'name' attribute
                    rewired_edge_list.append((int(G.vs[u]["name"]), int(G.vs[v]["name"])))
                null_pair_contacts[t] = rewired_edge_list
            else:
                null_pair_contacts[t] = aggregate_pair_contacts[t]

    return null_pair_contacts

# if __name__ == '__main__':
#     start_timestamp, end_timestamp = -1, -1
#     separator = ' '
#     mname = None
#
#     # Email EU, 500+ days in 45 mil. seconds
#     fname = "../tnet_sources/email-EU/email-Eu-core-temporal-Dept1.txt"
#     timestamp_first = False
#     period, time_label = 1, 'd'
#     start_timestamp, end_timestamp, add_missing = -1, -1, False
#     aggregate_time_to, strength = 86400, 0 # day, weak aggregation
#     min_cluster = 2
#
#     # Primary school, 2 days
#     # fname = "../tnet_sources/sociopatterns/co-presence/tij_pres_LyonSchool.dat"
#     # mname = "../tnet_sources/sociopatterns/metadata/metadata_LyonSchool.dat"
#     # timestamp_first = True
#     # period, time_label = 20, 's'
#     # start_timestamp, end_timestamp, add_missing = 120800, 151960, True
#     # aggregate_time_to, strength = 900, 0.5
#     # min_cluster = 2
#
#     # strings needed to name the output files
#     net_name = (fname.split("/")[-1]).split(".")[0]
#     suffix = "-min_" + str(min_cluster) + "-aggr_to_" + str(aggregate_time_to) + "-strength_" + str(strength)
#     if start_timestamp >= 0 and end_timestamp >= 0:
#         suffix += "-from_" + str(start_timestamp) + "_to_" + str(end_timestamp)
#
#     # the input data
#     pair_contacts = read_pair_contacts_from_file(fname, separator=separator, grain_t=period,
#                                                         start_timestamp=start_timestamp, end_timestamp=end_timestamp,
#                                                         timestamp_first=timestamp_first, add_missing=add_missing, verbose=False)
#     aggregate_pair_contacts = strongly_aggregate_time(pair_contacts,
#                                                         old_period=period, new_period=aggregate_time_to,
#                                                         strength=strength)
#     null_pair_contacts = null_model(aggregate_pair_contacts)

