def read_pair_contacts_from_file(fin_name, grain_t, start_timestamp=-1, end_timestamp=-1, verbose=False):
    """ Reads a temporal networks from a text file.

    Parameters
    ----------
        fin_name : str
            input file with whitespace-separated triplets per line:
            timestamp node1 node2 (assumed to appear chronologically)
        grain_t : int
            sensing interval in fin_name
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

    pair_contacts = {} # {timestamp : list of tuples (pair in contact)}
    timestamps = []

    with open(fin_name, 'r') as fin:
        for l in fin:
            t, u, v = map(int, l.split())

            if (start_timestamp >= 0 and end_timestamp >= 0 and t >= start_timestamp and t <= end_timestamp) or \
                    (start_timestamp < 0 and end_timestamp < 0):
                timestamps.append(t)
                lc = pair_contacts.get(t, [])
                lc.append(tuple(sorted([u, v]))) # ensures contacts are saved in a unique form
                pair_contacts[t] = lc
            elif end_timestamp >= 0 and t > end_timestamp:
                break

    # add missing timestamps (without contacts)
    timestamps = sorted(timestamps)
    for i in range(1, len(timestamps)):
        this_t = timestamps[i]
        prev_t = timestamps[i-1]
        if this_t > prev_t + grain_t:
            for missing_t in range(prev_t+grain_t, this_t, grain_t):
                #if missing_t not in pair_contacts: # not clear why needed
                pair_contacts[missing_t] = []

    if verbose:
        print("*** Pair contacts in", fin_name)
        tt = sorted(pair_contacts.keys())
        for t in tt:
            print("\t", t, pair_contacts[t])

    return pair_contacts

def read_node_metadata_from_file(fin_name, verbose=True):
    node_metadata = {} # {node id: metadata/category as string}
    categories = []

    with open(fin_name, 'r') as fin:
        for l in fin:
            node, metadata = l.split()
            node = int(node)
            node_metadata[node] = metadata
            categories.append(metadata)

    categories = list(set(categories))
    print(sorted(categories))

    return node_metadata

def initialise_aggregation(pair_contacts, old_period, new_period):
    old_timestamps = sorted(list(pair_contacts.keys()))
    aggregate_pair_contacts = {}

    # old: a contact was active in (t–old_period, t]
    # new: a contact  is active in (t–new_period, t]

    i = 0 # index in old timestamps
    new_timestamp = old_timestamps[0] - old_period + new_period

    while i < len(old_timestamps):
        while i < len(old_timestamps) and old_timestamps[i] <= new_timestamp:
            contacts = aggregate_pair_contacts.get(new_timestamp, [])
            contacts.extend(pair_contacts[old_timestamps[i]])
            aggregate_pair_contacts[new_timestamp] = contacts
            i += 1
        new_timestamp += new_period

    return aggregate_pair_contacts

# def weakly_aggregate_time(pair_contacts, old_period, new_period):
# 	aggregate_pair_contacts = initialise_aggregation(pair_contacts, old_period, new_period)

# 	# removes any duplicate contacts
# 	for t in aggregate_pair_contacts:
# 		aggregate_pair_contacts[t] = list(set(aggregate_pair_contacts[t]))

# 	return aggregate_pair_contacts

def strongly_aggregate_time(pair_contacts, old_period, new_period, strength=0.5):
    aggregate_pair_contacts = initialise_aggregation(pair_contacts, old_period, new_period)

    # removes all contacts from an aggregated timestamp which occur too infrequently
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

def normalise_list_pair_contacts(pair_contacts, node_metadata=None, old_time_unit='s'):

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
    if len(str(timestamps[0])) == 10: # timestamps are Unix? convert to string in my local time zone
        for i in timestamp_reverse_translator:
            t = timestamp_reverse_translator[i]
            timestamp_reverse_translator[i] = strftime("%H:%M", gmtime(t))
    elif old_time_unit == 's':        # timestamps are seconds? convert to hour:minute
        for i in timestamp_reverse_translator:
            t = int(timestamp_reverse_translator[i])
            timestamp_reverse_translator[i] = str(int(t/3600)%24) + ":" + str(int((t%3600)/60)).zfill(2)

    # node id translator from original node id to a 0-index, and back
    nodeid_translator = dict([(old, new) for (new, old) in enumerate(nodeids)])
    nodeid_reverse_translator = dict(enumerate(nodeids))

    # node metadata as a list following the new node ids
    node_metadata_list = []
    if node_metadata:
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

if __name__ == '__main__':
    start_timestamp, end_timestamp = -1, -1
    period = 20

    # fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LH10.dat"
    # fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_07_17.txt"
    # original_time = 20
    # delta_time = 200
    # aggregate_time_to = 80

    # Primary school, 2 days
    fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LyonSchool.dat"
    mname = "tnet_sources/sociopatterns/metadata/metadata_LyonSchool.dat"
    start_timestamp, end_timestamp = 120800, 130000 #151960
    aggregate_time_to, strength = 600, 0.5

    pair_contacts = read_pair_contacts_from_file(fname, period, start_timestamp, end_timestamp, False)
    node_metadata = read_node_metadata_from_file(mname)
    print(node_metadata)

    aggregate_pair_contacts = strongly_aggregate_time(pair_contacts,
                                                      old_period=period, new_period=aggregate_time_to,
                                                      strength=strength)

    normalised_list_pair_contacts, num_timestamps, num_nodeids, timestamp_reverse_translator, nodeid_translator, node_metadata_list = \
        normalise_list_pair_contacts(aggregate_pair_contacts, node_metadata)
    print(node_metadata_list)
