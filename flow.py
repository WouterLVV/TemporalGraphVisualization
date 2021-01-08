from tgv.timegraph import TimeGraph
from tgv.layout import SizedConnectionLayout

from tgv.io_operations import *
from tgv.colours import *

start_timestamp, end_timestamp = -1, -1
separator = ' '
mname = None

# Hypertext conference, 2.5 days
# fname = "tnet_sources/sociopatterns-hypertext09/ht09_contact_list.dat"
# separator = '\t'
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, True
# aggregate_time_to = 180
# strength = 0
# min_cluster = 2

# Conference, 2 days
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_SFHH.dat"
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, True
# aggregate_time_to = 1200
# strength = 0.5
# min_cluster = 2

# Hospital ward, 3 days in a continuous block
fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LH10.dat"
timestamp_first = True
period, time_label = 20, 's'
start_timestamp, end_timestamp, add_missing = -1, -1, True
aggregate_time_to = 1080
strength = 0.5
min_cluster = 2

# Workplace, 2 weeks x 8am-
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_InVS13.dat"
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_InVS15.dat"
# aggregate_time_to, strength = 600, 0.5

# Science Gallery
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_28.txt"
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_29.txt"
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_07_17.txt"
# aggregate_time_to, strength = 120, 0.5

# Primary school, 2 days
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LyonSchool.dat"
# mname = "tnet_sources/sociopatterns/metadata/metadata_LyonSchool.dat"
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = 120800, 151960, True
# aggregate_time_to, strength = 1020, 0.5
# min_cluster = 2

# High school, 5 days x 8am-6pm
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_Thiers13.dat"
# mname = "tnet_sources/sociopatterns/metadata/metadata_Thiers13.dat"
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = 29960, 64780, True
# aggregate_time_to = 600
# strength = 0.75
# min_cluster = 4

# Science Gallery
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_28.txt"
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_29.txt"
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_07_17.txt"
# aggregate_time_to, strength = 120, 0.5
# colormap = {}

# Email EU, 500+ days in 45 mil. seconds
# fname = "tnet_sources/email-EU/email-Eu-core-temporal-Dept1.txt"
# timestamp_first = False
# period, time_label = 1, 'd'
# start_timestamp, end_timestamp, add_missing = -1, -1, False
# aggregate_time_to, strength = 1857600, 1/86400 # day, weak aggregation
# min_cluster = 2

# College msg, 193 days
# fname = "tnet_sources/college-msg/CollegeMsg.txt"
# timestamp_first = False
# period, time_label = 1, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, False
# aggregate_time_to = 216000 # 86400 # day
# strength = 1/86400
# min_cluster = 4

# Copenhagen smses, 27 days
# fname = "tnet_sources/copenhagen-study/sms.csv"
# separator = ','
# timestamp_first = True
# period, time_label = 1, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, True # 0, 128018, True
# aggregate_time_to = 2700 # 3/4 hour
# strength = 0
# min_cluster = 2

if __name__ == '__main__':

    # strings needed to name the output files
    net_name = (fname.split("/")[-1]).split(".")[0]
    suffix = "-min_" + str(min_cluster) + "-aggr_to_" + str(aggregate_time_to) + "-strength_" + str(strength)
    if start_timestamp >= 0 and end_timestamp >= 0:
        suffix += "-from_" + str(start_timestamp) + "_to_" + str(end_timestamp)

    # the input data
    pair_contacts = read_pair_contacts_from_file(fname, separator=separator, grain_t=period,
                                                        start_timestamp=start_timestamp, end_timestamp=end_timestamp,
                                                        timestamp_first=timestamp_first, add_missing=add_missing, verbose=False)
    node_metadata, colormap = None, None
    if mname:
        node_metadata, categories = read_node_metadata_from_file(mname)
        colormap = assign_colours_rgba_tuple(categories)

    # input data preprocessing
    aggregate_pair_contacts = strongly_aggregate_time(pair_contacts,
                                                      old_period=period, new_period=aggregate_time_to,
                                                      strength=strength)

    # transform data to a null model?
    null_pair_contacts = null_model(aggregate_pair_contacts)

    normalised_list_pair_contacts, num_timestamps, num_nodeids, timestamp_reverse_translator, nodeid_translator, node_metadata_list = \
                              normalise_list_pair_contacts(aggregate_pair_contacts, node_metadata, time_label=time_label)

    node_list = node_metadata_list
    if not node_metadata_list:
        node_list = num_nodeids

    # data to temporal graph
    g = TimeGraph(normalised_list_pair_contacts, node_list, num_timestamps, minimum_cluster_size=min_cluster, minimum_connection_size=min_cluster)

    sg = SizedConnectionLayout(g, line_width=1,
                               cluster_height_method='linear',
                               horizontal_density=1,
                               vertical_density=0.5)

    sg.set_order(barycenter_passes=10)
    sg.set_alignment(stairs_iterations=5)
    sg.draw_graph(filename="flow_output/"+net_name+suffix+".svg",
                  colormap=colormap,
                  timestamp_translator=timestamp_reverse_translator,
                  show_annotations=False,
                  show_timestamps=True,
                  stats_info=("homogeneity_diff", "homogeneity", "in_out_difference", "layer_num_clusters", "layer_num_members"),
                  fading=True)
    print(net_name+suffix)
