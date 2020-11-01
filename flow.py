from TimeGraph import TimeGraph
from layout import SugiyamaLayout
import csv

from io_operations import *

start_timestamp, end_timestamp = -1, -1
period = 20

# Hypertext conference, 2.5 days
# fname = "tnet_sources/sociopatterns-hypertext09/ht09_contact_list.dat"
# aggregate_time_to, strength = 120, 0

# Conference, 2 days
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_SFHH.dat"
# aggregate_time_to, strength = 300, 0.5

# Hospital ward, 3 days in a continuous block
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LH10.dat"
# aggregate_time_to, strength = 600, 0.5

# Workplace, 2 weeks x 8am-
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_InVS13.dat"
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_InVS15.dat"
# aggregate_time_to, strength = 600, 0.5

# Primary school, 2 days
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LyonSchool.dat"
# mname = "tnet_sources/sociopatterns/metadata/metadata_LyonSchool.dat"
# colormap = {'teachers': (1,0,0,1), 'cm1a': (0,1,0,1), 'cm1b': (0,0,1,1), 'cm2a': (1,0.5,0,1), 'cm2b': (0,0.5,1,1), 'ce1a': (0.5,0,1,1), 'ce1b': (1,1,0,1), 'ce2a': (1,0,1,1), 'ce2b': (0,1,1,1), 'cpa': (0.3,0.3,0.3,1), 'cpb': (0.6,0.6,0.6,1)}
# start_timestamp, end_timestamp = 120800, 151960
# aggregate_time_to, strength = 600, 0.5

# High school, 5 days x 8am-6pm
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_Thiers13.dat"
# mname = "tnet_sources/sociopatterns/metadata/metadata_Thiers13.dat"
# colormap = {'2BIO1': (1,0,0,1), '2BIO2': (0,1,0,1), '2BIO3': (0,0,1,1), 'MP': (1,0.5,0,1), 'MPst1': (0,0.5,1,1), 'MPst2': (0.5,0,1,1), 'PC': (1,1,0,1), 'PCst': (1,0,1,1), 'PSIst': (0,1,1,1)}
# start_timestamp, end_timestamp = 29960, 64780
# aggregate_time_to, strength = 600, 0.5

# Science Gallery
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_28.txt"
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_29.txt"
fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_07_17.txt"
aggregate_time_to, strength = 120, 0.5
colormap = {}

net_name = (fname.split("/")[-1]).split(".")[0]
suffix = ""
if start_timestamp >= 0 and end_timestamp >= 0:
    suffix = "-from_" + str(start_timestamp) + "_to_" + str(end_timestamp)

if __name__ == '__main__':
    pair_contacts = read_pair_contacts_from_file(fname, period, start_timestamp, end_timestamp, False)
    node_metadata = None
    # node_metadata = read_node_metadata_from_file(mname)

    aggregate_pair_contacts = strongly_aggregate_time(pair_contacts, 
                                                      old_period=period, new_period=aggregate_time_to, 
                                                      strength=strength)

    normalised_list_pair_contacts, num_timestamps, num_nodeids, timestamp_reverse_translator, nodeid_translator, node_metadata_list = \
                              normalise_list_pair_contacts(aggregate_pair_contacts, node_metadata)

    g = TimeGraph(normalised_list_pair_contacts, num_nodeids, num_timestamps)
    sg = SugiyamaLayout(g, minimum_cluster_size=2, minimum_connection_size=2,
                           horizontal_density=2,
                           vertical_density=1)
    sg.draw_graph(filename="flow_output/"+net_name+suffix+".svg",
                  colormap=colormap,
                  timestamp_translator=timestamp_reverse_translator)
