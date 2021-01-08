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
# aggregate_time_to = 120
# aggregate_time_to_range = list(range(60, 1800, 20))
# strength = 0
# min_cluster = 2

# Conference, 2 days
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_SFHH.dat"
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, True
# aggregate_time_to = 300
# aggregate_time_to_range = list(range(60, 3000, 20))
# strength = 0.5
# min_cluster = 2

# Hospital ward, 3 days in a continuous block
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LH10.dat"
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, True
# aggregate_time_to = 600
# aggregate_time_to_range = list(range(60, 3000, 20))
# strength = 0.5
# min_cluster = 2

# Workplace, 2 weeks x 8am-
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_InVS13.dat"
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_InVS15.dat"
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, True
# aggregate_time_to = 3600
# aggregate_time_to_range = list(range(60, 3600, 60))
# strength = 0.5
# min_cluster = 2

# Science Gallery
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_28.txt"
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_29.txt"
fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_07_17.txt"
separator = '\t'
timestamp_first = True
period, time_label = 20, 's'
start_timestamp, end_timestamp, add_missing = -1, -1, True
aggregate_time_to = 120
aggregate_time_to_range = list(range(60, 1200, 20))
strength = 0.5
min_cluster = 2

# Primary school, 2 days
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LyonSchool.dat"
# mname = "tnet_sources/sociopatterns/metadata/metadata_LyonSchool.dat"
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = 120800, 151960, True
# aggregate_time_to = 60
# aggregate_time_to_range = list(range(60, 1800, 20))
# strength = 0.5
# min_cluster = 2

# High school, 5 days x 8am-6pm
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_Thiers13.dat"
# mname = "tnet_sources/sociopatterns/metadata/metadata_Thiers13.dat"
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = 29960, 64780, True
# aggregate_time_to = 600
# aggregate_time_to_range = list(range(60, 3600, 20))
# strength = 0.5
# min_cluster = 5

# Email EU, 500+ days in 45 mil. seconds
# fname = "tnet_sources/email-EU/email-Eu-core-temporal-Dept1.txt"
# timestamp_first = False
# period, time_label = 1, 'd'
# start_timestamp, end_timestamp, add_missing = -1, -1, False
# aggregate_time_to = 7*86400 # week
# aggregate_time_to_range = list(range(3600, 10*86400, 3600))
# strength = 1/86400
# min_cluster = 2

# College msg, 193 days
# fname = "tnet_sources/college-msg/CollegeMsg.txt"
# timestamp_first = False
# period, time_label = 1, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, False
# aggregate_time_to = 86400 # day
# aggregate_time_to_range = list(range(4*3600, 14*86400, 4*3600))
# strength = 0
# min_cluster = 4

# Copenhagen smses, 27 days
# fname = "tnet_sources/copenhagen-study/sms.csv"
# separator = ','
# timestamp_first = True
# period, time_label = 1, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, True
# aggregate_time_to = 3600
# aggregate_time_to_range = list(range(900, 43200, 900))
# strength = 0
# min_cluster = 2

def build_temporal_graph(normalised_list_pair_contacts, node_list, num_timestamps, min_cluster, net_name, suffix, colormap, timestamp_reverse_translator, draw=False):
    g = TimeGraph(normalised_list_pair_contacts, node_list, num_timestamps, minimum_cluster_size=min_cluster, minimum_connection_size=min_cluster)

    # draw
    if draw:
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
                      fading=True)

    return g

def get_net_name(fname):
    return (fname.split("/")[-1]).split(".")[0]

def get_suffix(min_cluster, aggregate_time_to, strength, start_timestamp, end_timestamp):
    suffix = "-min_" + str(min_cluster) + "-aggr_to_" + str(aggregate_time_to) + "-strength_" + str(strength)
    if start_timestamp >= 0 and end_timestamp >= 0:
        suffix += "-from_" + str(start_timestamp) + "_to_" + str(end_timestamp)

    return suffix

def form_node_list(node_metadata_list, num_nodeids):
    node_list = node_metadata_list
    if not node_metadata_list:
        node_list = num_nodeids

    return node_list

def stability_metric(start_count, end_count, merge_count, split_count, stable_count):
    if split_count > 0:
        return stable_count / (split_count)
    return 0

def stat_plot(ylabel, res):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(aggregate_time_to_range, res)
    plt.scatter(aggregate_time_to_range, res)
    plt.xlabel(xlabel)
    plt.xticks(ticks=[0]+aggregate_time_to_range[5::6], rotation=270)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == '__main__':

    # the input data
    net_name = get_net_name(fname)
    print("***", net_name)
    pair_contacts = read_pair_contacts_from_file(fname, separator=separator, grain_t=period,
                                                        start_timestamp=start_timestamp, end_timestamp=end_timestamp,
                                                        timestamp_first=timestamp_first, add_missing=add_missing, verbose=False)
    node_metadata, colormap = None, None
    if mname:
        node_metadata, categories = read_node_metadata_from_file(mname)
        colormap = assign_colours_rgba_tuple(categories)

    # grid test across one parameter
    differential_density = []
    differential_stability = []

    for aggregate_time_to in aggregate_time_to_range:

        # the string needed to name the output files
        suffix = get_suffix(min_cluster, aggregate_time_to, strength, start_timestamp, end_timestamp)

        # the input data
        aggregate_pair_contacts = strongly_aggregate_time(pair_contacts, old_period=period, new_period=aggregate_time_to, strength=strength)

        # ... the temporal graph for the input data
        normalised_list_pair_contacts, num_timestamps, num_nodeids, timestamp_reverse_translator, nodeid_translator, node_metadata_list = \
                    normalise_list_pair_contacts(aggregate_pair_contacts, node_metadata, time_label=time_label)
        node_list = form_node_list(node_metadata_list, num_nodeids)
        g =         build_temporal_graph(normalised_list_pair_contacts, node_list, num_timestamps, 
                    min_cluster, net_name, suffix, colormap, timestamp_reverse_translator, draw=False)

        start_count, end_count, merge_count, split_count, stable_count = g.num_events()
        g_density = g.density()
        g_stability = stability_metric(start_count, end_count, merge_count, split_count, stable_count)

        # the null model, bootstrapped
        g_null_density_bootstrapped = []
        g_null_stability_bootstrapped = []
        from numpy import mean

        for i in range(1):
            null_pair_contacts = null_model(aggregate_pair_contacts)

            # ... the temporal graph for the null model
            normalised_list_pair_contacts, num_timestamps, num_nodeids, timestamp_reverse_translator, nodeid_translator, node_metadata_list = \
                        normalise_list_pair_contacts(null_pair_contacts, node_metadata, time_label=time_label)
            node_list = form_node_list(node_metadata_list, num_nodeids)
            g_null =    build_temporal_graph(normalised_list_pair_contacts, node_list, num_timestamps, 
                        min_cluster, net_name, suffix+"-null_model", colormap, timestamp_reverse_translator, draw=False)

            g_null_density_bootstrapped.append(g_null.density())
            start_count, end_count, merge_count, split_count, stable_count = g_null.num_events()
            g_null_stability_bootstrapped.append(stability_metric(start_count, end_count, merge_count, split_count, stable_count))

        g_null_density = mean(g_null_density_bootstrapped)
        g_null_stability = mean(g_null_stability_bootstrapped)

        # the differential statistics between the two temporal graphs
        res = g_density - g_null_density
        # print("  -->", suffix, "\t", g_density, "\t", g_null_density, "\t", "diff:", res)
        differential_density.append(res)

        res = g_stability # - g_null_stability
        print("  -->", suffix, "\t", g_stability, "\t", g_null_stability, "\t", "diff:", res)
        differential_stability.append(res)

    # plot the grid test
    xlabel = "Aggregation period (s)"
    stat_plot("Stability", differential_stability)
