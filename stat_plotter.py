import matplotlib.pyplot as plt
from TimeGraph import TimeGraph
from layout import SugiyamaLayout

from io_operations import *

start_timestamp, end_timestamp = -1, -1
separator = ' '
mname = None

# Primary school, 2 days
fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LyonSchool.dat"
# mname = "tnet_sources/sociopatterns/metadata/metadata_LyonSchool.dat"
timestamp_first = True
period, time_label = 20, 's'
start_timestamp, end_timestamp, add_missing = 120800, 151960, True
strength = 0.5
min_cluster = 2
evaluate_at = [i for i in range(20, 2000, 40)]

# Email EU, 500+ days in 45 mil. seconds
# fname = "tnet_sources/email-EU/email-Eu-core-temporal-Dept1.txt"
# timestamp_first = False
# period, time_label = 1, 'd'
# start_timestamp, end_timestamp, add_missing = -1, -1, False
# strength = 0 # day, weak aggregation
# min_cluster = 2
# evaluate_at = [i for i in range(21600, 21600*29, 21600)]  # steps of 6 hours up to 1 week

if __name__ == "__main__":

    net_name = (fname.split("/")[-1]).split(".")[0]

    pair_contacts = read_pair_contacts_from_file(fname, separator=separator, grain_t=period,
                                                 start_timestamp=start_timestamp, end_timestamp=end_timestamp,
                                                 timestamp_first=timestamp_first, add_missing=add_missing,
                                                 verbose=False)
    results = []
    ylabel = ""
    for aggregate_time_to in evaluate_at:

        aggregate_pair_contacts = strongly_aggregate_time(pair_contacts,
                                                          old_period=period, new_period=aggregate_time_to,
                                                          strength=strength)

        normalised_list_pair_contacts, num_timestamps, num_nodeids, timestamp_reverse_translator, nodeid_translator, node_metadata_list = \
            normalise_list_pair_contacts(aggregate_pair_contacts, None, time_label=time_label)

        node_list = num_nodeids

        g = TimeGraph(normalised_list_pair_contacts, node_list, num_timestamps, minimum_cluster_size=min_cluster,
                      minimum_connection_size=min_cluster)


        # ylabel = "number of clusters"
        # results.append(g.num_clusters())

        # ylabel = "Average relative continuity"
        # results.append(g.average_relative_continuity())

        # ylabel = "Average absolute continuity"
        # results.append(g.average_absolute_continuity())

        # ylabel = "Average relative continuity difference"
        # results.append(g.average_relative_continuity_diff())

        # ylabel = "Average absolute continuity difference"
        # results.append(g.average_absolute_continuity_diff())

        # ylabel = "Normalized absolute continuity"
        # results.append(g.normalized_absolute_continuity())

        # ylabel = "Normalized absolute continuity difference"
        # results.append(g.normalized_absolute_continuity_diff())

    xlabel = "Aggregation"
    plt.figure()
    plt.plot(evaluate_at, results)
    plt.scatter(evaluate_at, results)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()