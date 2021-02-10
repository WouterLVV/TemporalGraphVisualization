import os
from copy import copy

from tgv.layout import SizedConnectionLayout
from tgv.timegraph import TimeGraph
from tgv.io_operations import read_pair_contacts_from_file, read_node_metadata_from_file, add_missing_timestamps, strongly_aggregate_time, node_ids_from_pair_contacts, normalise_timestamps

class ImportSettings:
    def __init__(self, filename, metafilename=None, separator=' ',
                 timestamp_first=True, file_period=-1, time_label='s',
                 start_timestamp=-1, end_timestamp=-1, add_missing=True,
                 colormap=None, minimum_cluster_size=2, minimum_connection_size=2,
                 period=-1, agg_strength=0.5):

        self.filename = filename
        self.metafilename = metafilename  # Can be None for no metadata
        self.separator = separator
        self.timestamp_first = timestamp_first
        self.file_period = file_period  # The period of the data in the file, by default it will be automatically detected in DataContainer
        self.timelabel = time_label
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.add_missing = add_missing
        self.colormap = colormap  # if None will generate automatically
        self.minimum_connection_size = minimum_connection_size
        self.minimum_cluster_size = max(minimum_cluster_size, minimum_connection_size)
        self.period = period  # The period to aggregate data to (-1 to not aggregate)
        self.agg_strength = agg_strength  # strength to use in aggregation

        self.hasmeta = self.metafilename is not None


class DataContainer:
    def __init__(self, settings: ImportSettings):
        self.settings = settings
        self.pair_contacts = read_pair_contacts_from_file(self.settings.filename, self.settings.separator,
                                                          self.settings.start_timestamp, self.settings.end_timestamp,
                                                          self.settings.timestamp_first, verbose=False)

        self.node_metadata, self.node_categories = None, None
        if self.settings.hasmeta:
            self.node_metadata, self.node_categories = read_node_metadata_from_file(self.settings.metafilename)

        if settings.file_period <= 0:
            settings.file_period = self.infer_period()

        if settings.start_timestamp < 0:
            settings.start_timestamp = min(self.pair_contacts.keys())

        if settings.end_timestamp < 0:
            settings.end_timestamp = max(self.pair_contacts.keys())

        if settings.period > settings.file_period:
            agg_to = settings.period  # Newly wanted period
            settings.period = settings.file_period  # Function expects current period in settings.period
            self.aggregate_to(agg_to, settings.agg_strength)
        else:
            settings.period = settings.file_period

        if self.settings.add_missing:
            add_missing_timestamps(self.pair_contacts, self.settings.period,
                                   self.settings.start_timestamp, self.settings.end_timestamp, verbose=False)

        if self.settings.hasmeta:
            if self.settings.colormap is None:
                self.auto_color()

        self.net_name = (settings.filename.split(os.sep)[-1]).split(".")[0]
        self.net_suffix = "-min_" + str(self.settings.minimum_cluster_size) \
                          + "-aggr_to_" + str(self.settings.period) \
                          + "-strength_" + str(self.settings.agg_strength)

        self.timestamp_reverse_translator = None  # Generated when get_timegraph is called

    def infer_period(self):
        from math import gcd
        tt = list(self.pair_contacts.keys())
        if len(tt) < 2:
            return 1
        x = tt[1] - tt[0]
        for i in range(2, len(tt)):
            x = gcd(x, tt[i] - tt[i-1])
            if x == 1:
                break
        return x

    def add_metadata(self, node_metadata, update_colormap=True):
        self.settings.hasmeta = True
        self.node_metadata = node_metadata
        self.node_categories = sorted(list(set(node_metadata.values())))
        if update_colormap:
            self.auto_color()

    def auto_color(self):
        from tgv.colours import assign_colours_rgba_tuple
        self.settings.colormap = assign_colours_rgba_tuple(self.node_categories)

    def duration(self):
        return str(self.settings.end_timestamp - self.settings.start_timestamp) + self.settings.timelabel

    def get_num_steps(self):
        return len(self.pair_contacts.keys())

    def get_timegraph(self):
        time_normalised_pair_contacts, timestamp_reverse_translator = normalise_timestamps(self.pair_contacts, self.settings.timelabel)
        self.timestamp_reverse_translator = timestamp_reverse_translator
        node_ids = node_ids_from_pair_contacts(self.pair_contacts)

        if self.settings.hasmeta:
            node_ids = {node_id: self.node_metadata.get(node_id, None) for node_id in node_ids}

        tg = TimeGraph(time_normalised_pair_contacts, node_ids, self.get_num_steps(),
                       self.settings.minimum_cluster_size, self.settings.minimum_connection_size)

        return tg

    def get_SCLayout(self):
        tg = self.get_timegraph()
        sc = SizedConnectionLayout(tg)
        return sc

    def draw_graph(self, output=""):
        if output == "":
            output = "flow_output/" + self.net_name + self.net_suffix + ".svg"
        elif output[-1] == os.sep:
            output = output + self.net_name + self.net_suffix + ".svg"

        sc = self.get_SCLayout()
        sc.draw_graph(filename=output, colormap=self.settings.colormap,
                      show_timestamps=True, timestamp_translator=self.timestamp_reverse_translator)

    def aggregate_to(self, period, strength=-1., makecopy=False):
        if makecopy:  # SHALLOW COPY! Is safe to generate different aggregations, since the Timegraph only reads.
            container = copy(self)
            container.settings = copy(self.settings)
        else:
            container = self

        if strength < 0:
            strength = self.settings.agg_strength
        else:
            self.settings.agg_strength = strength

        container.pair_contacts = strongly_aggregate_time(container.pair_contacts, container.settings.period, period, strength)
        container.settings.period = period

        if container.settings.add_missing:
            add_missing_timestamps(container.pair_contacts, container.settings.period,
                                   container.settings.start_timestamp, container.settings.end_timestamp, verbose=False)

        # Update suffix name
        self.net_suffix = "-min_" + str(self.settings.minimum_cluster_size) \
                          + "-aggr_to_" + str(self.settings.period) \
                          + "-strength_" + str(self.settings.agg_strength)
        return container

def copy_and_change_period_to(settings, newperiod):
    c = copy(settings)
    c.period = newperiod
    return c


LYONSCHOOL_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns/co-presence/tij_pres_LyonSchool.dat",
    metafilename="tnet_sources/sociopatterns/metadata/metadata_LyonSchool.dat",
    start_timestamp=120800, end_timestamp=151960,
    minimum_connection_size=5
)

LYONSCHOOL_SETTINGS_AGG60 = copy_and_change_period_to(LYONSCHOOL_SETTINGS_BASE, 60)  # 1 minute
LYONSCHOOL_SETTINGS_AGG300 = copy_and_change_period_to(LYONSCHOOL_SETTINGS_BASE, 300)  # 5 minutes
LYONSCHOOL_SETTINGS_AGG600 = copy_and_change_period_to(LYONSCHOOL_SETTINGS_BASE, 600)  # 10 minutes
LYONSCHOOL_SETTINGS_AGG900 = copy_and_change_period_to(LYONSCHOOL_SETTINGS_BASE, 900)  # 15 minutes
LYONSCHOOL_SETTINGS_AGG3600 = copy_and_change_period_to(LYONSCHOOL_SETTINGS_BASE, 3600)  # 1 hour

LYONSCHOOL_SETTINGS_DEFAULT = LYONSCHOOL_SETTINGS_AGG600

LYONSCHOOL_EVALUATE_RANGE = range(20, 2000, 80)


EMAILEU_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/email-EU/email-Eu-core-temporal-Dept1.txt",
    timestamp_first=False,
    add_missing=False,
    agg_strength=1/86400,
    time_label='d'
)

EMAILEU_SETTINGS_12HOUR = copy_and_change_period_to(EMAILEU_SETTINGS_BASE, 86400/2)
EMAILEU_SETTINGS_DAY = copy_and_change_period_to(EMAILEU_SETTINGS_BASE, 86400)
EMAILEU_SETTINGS_2DAYS = copy_and_change_period_to(EMAILEU_SETTINGS_BASE, 86400*2)
EMAILEU_SETTINGS_4DAYS = copy_and_change_period_to(EMAILEU_SETTINGS_BASE, 86400*4)
EMAILEU_SETTINGS_WEEK = copy_and_change_period_to(EMAILEU_SETTINGS_BASE, 86400*7)

EMAILEU_SETTINGS_DEFAULT = EMAILEU_SETTINGS_DAY


HYPERTEXTCONFERENCE_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns-hypertext09/ht09_contact_list.dat",
    separator='\t',
    agg_strength=0
)

HYPERTEXTCONFERENCE_SETTINGS_AGG120 = copy_and_change_period_to(HYPERTEXTCONFERENCE_SETTINGS_BASE, 120)

HYPERTEXTCONFERENCE_SETTINGS_DEFAULT = HYPERTEXTCONFERENCE_SETTINGS_AGG120


SFHHCONFERENCE_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns/co-presence/tij_pres_SFHH.dat",
)

SFHHCONFERENCE_SETTINGS_AGG300 = copy_and_change_period_to(SFHHCONFERENCE_SETTINGS_BASE, 300)

SFHHCONFERENCE_SETTINGS_DEFAULT = SFHHCONFERENCE_SETTINGS_AGG300


HOSPITALWARD_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns/co-presence/tij_pres_LH10.dat"
)

HOSPITALWARD_SETTINGS_AGG600 = copy_and_change_period_to(HOSPITALWARD_SETTINGS_BASE, 600)

HOSPITALWARD_SETTINGS_DEFAULT = HOSPITALWARD_SETTINGS_AGG600


WORKPLACE13_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns/co-presence/tij_pres_InVS13.dat"
)

WORKPLACE13_SETTINGS_AGG3600 = copy_and_change_period_to(WORKPLACE13_SETTINGS_BASE, 3600)

WORKPLACE13_SETTINGS_DEFAULT = WORKPLACE13_SETTINGS_AGG3600


WORKPLACE15_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns/co-presence/tij_pres_InVS15.dat"
)

WORKPLACE15_SETTINGS_AGG3600 = copy_and_change_period_to(WORKPLACE15_SETTINGS_BASE, 3600)

WORKPLACE15_SETTINGS_DEFAULT = WORKPLACE15_SETTINGS_AGG3600


SCIENCEGALLERY0428_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns-infectious/listcontacts_2009_04_28.txt",
    separator='\t'
)

SCIENCEGALLERY0428_SETTINGS_AGG120 = copy_and_change_period_to(SCIENCEGALLERY0428_SETTINGS_BASE, 120)

SCIENCEGALLERY0428_SETTINGS_DEFAULT = SCIENCEGALLERY0428_SETTINGS_AGG120


SCIENCEGALLERY0429_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns-infectious/listcontacts_2009_04_29.txt",
    separator='\t'
)

SCIENCEGALLERY0429_SETTINGS_AGG120 = copy_and_change_period_to(SCIENCEGALLERY0429_SETTINGS_BASE, 120)

SCIENCEGALLERY0429_SETTINGS_DEFAULT = SCIENCEGALLERY0429_SETTINGS_AGG120


SCIENCEGALLERY0717_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns-infectious/listcontacts_2009_07_17.txt",
    separator='\t'
)

SCIENCEGALLERY0717_SETTINGS_AGG120 = copy_and_change_period_to(SCIENCEGALLERY0717_SETTINGS_BASE, 120)

SCIENCEGALLERY0717_SETTINGS_DEFAULT = SCIENCEGALLERY0717_SETTINGS_AGG120

SCIENCEGALLERY_EVALUATE_RANGE = range(60, 1200, 20)


THIERSSCHOOL_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/sociopatterns/co-presence/tij_pres_Thiers13.dat",
    metafilename="tnet_sources/sociopatterns/metadata/metadata_Thiers13.dat",
    start_timestamp=29960, end_timestamp=64780,
    minimum_connection_size=5
)

THIERSSCHOOL_SETTINGS_AGG600 = copy_and_change_period_to(THIERSSCHOOL_SETTINGS_BASE, 600)

THIERSSCHOOL_SETTINGS_DEFAULT = THIERSSCHOOL_SETTINGS_AGG600


COLLEGEMSG_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/college-msg/CollegeMsg.txt",
    timestamp_first=False,
    agg_strength=0,
    minimum_connection_size=4
)

COLLEGEMSG_SETTINGS_12HOUR = copy_and_change_period_to(COLLEGEMSG_SETTINGS_BASE, 86400/2)
COLLEGEMSG_SETTINGS_DAY = copy_and_change_period_to(COLLEGEMSG_SETTINGS_BASE, 86400)
COLLEGEMSG_SETTINGS_2DAYS = copy_and_change_period_to(COLLEGEMSG_SETTINGS_BASE, 86400*2)
COLLEGEMSG_SETTINGS_4DAYS = copy_and_change_period_to(COLLEGEMSG_SETTINGS_BASE, 86400*4)
COLLEGEMSG_SETTINGS_WEEK = copy_and_change_period_to(COLLEGEMSG_SETTINGS_BASE, 86400*7)

COLLEGEMSG_SETTINGS_DEFAULT = COLLEGEMSG_SETTINGS_DAY


COPENHAGENSMS_SETTINGS_BASE = ImportSettings(
    filename="tnet_sources/copenhagen-study/sms.csv",
    separator=',',
    agg_strength=0.
)

COPENHAGENSMS_SETTINGS_AGG3600 = copy_and_change_period_to(COPENHAGENSMS_SETTINGS_BASE, 3600)

COPENHAGENSMS_SETTINGS_DEFAULT = COPENHAGENSMS_SETTINGS_AGG3600


TESTSUITE_DEFAULTS = [
    LYONSCHOOL_SETTINGS_DEFAULT,
    EMAILEU_SETTINGS_DEFAULT,
    COPENHAGENSMS_SETTINGS_DEFAULT,
    WORKPLACE13_SETTINGS_DEFAULT,
    SCIENCEGALLERY0428_SETTINGS_DEFAULT,
    HYPERTEXTCONFERENCE_SETTINGS_DEFAULT,
    SFHHCONFERENCE_SETTINGS_DEFAULT,
    HOSPITALWARD_SETTINGS_DEFAULT
]


if __name__ == "__main__":
    # Example pair_contacts
    dc = DataContainer(EMAILEU_SETTINGS_DAY)
    print("Example pair_contacts:", len(dc.pair_contacts))

    # Example get timegraph
    dc = DataContainer(HOSPITALWARD_SETTINGS_DEFAULT)
    tg = dc.get_timegraph()
    print("Example timegraph:", tg.num_events())

    # Example get layout
    dc = DataContainer(WORKPLACE13_SETTINGS_AGG3600)
    sc = dc.get_SCLayout()
    print("Example layout:", sc.max_bundle_size)

    # example aggregate to arbitrary period
    dc = DataContainer(LYONSCHOOL_SETTINGS_BASE)  # Base has no pre-aggregation applied to it
    dc.aggregate_to(555)  # aggregate to this period size.
    # If makecopy is not used, overwrites the current period and pair_contacts and can't be reversed.
    print("Example aggregation:", dc.get_num_steps())

    # Example aggregate to different periods
    dc = DataContainer(LYONSCHOOL_SETTINGS_BASE)
    # Use makecopy to create a new container.
    # Be warned that it uses shallow copies for some variables,
    # for example adding an element to the colormap will place it in all copies.
    dc1 = dc.aggregate_to(111, makecopy=True)
    dc2 = dc.aggregate_to(222, strength=0.75, makecopy=True)  # passing a strength argument will overwrite the old strength setting (in the copy)
    dc3 = dc.aggregate_to(333, makecopy=True)
    print("Example multi aggregation:", dc2.settings.agg_strength, dc3.settings.agg_strength)

    # example testing a range of aggs
    # I recommend using the aggregate_to function of datacontainer, because this avoids rereading and parsing from disk
    dc = DataContainer(EMAILEU_SETTINGS_BASE)
    # agg day to week with 1 day interval and take the TimeGraph of each
    tgs = [dc.aggregate_to(x, makecopy=True).get_timegraph() for x in range(86400, 86400*7+1, 86400)]
    print("Example range aggregation:", [tg.num_clusters() for tg in tgs])

    # example generate graphs for entire testsuite with default setting along the entire chain
    for settings in TESTSUITE_DEFAULTS:
        DataContainer(settings).draw_graph()



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

# College msg, 193 days
# fname = "tnet_sources/college-msg/CollegeMsg.txt"
# timestamp_first = False
# period, time_label = 1, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, False
# aggregate_time_to = 86400 # day
# aggregate_time_to_range = list(range(4*3600, 14*86400, 4*3600))
# strength = 0
# min_cluster = 4


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


# Science Gallery
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_28.txt"
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_04_29.txt"
# fname = "tnet_sources/sociopatterns-infectious/listcontacts_2009_07_17.txt"
# separator = '\t'
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, True
# aggregate_time_to = 120
# aggregate_time_to_range = list(range(60, 1200, 20))
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

# Hospital ward, 3 days in a continuous block
# fname = "tnet_sources/sociopatterns/co-presence/tij_pres_LH10.dat"
# timestamp_first = True
# period, time_label = 20, 's'
# start_timestamp, end_timestamp, add_missing = -1, -1, True
# aggregate_time_to = 600
# aggregate_time_to_range = list(range(60, 3000, 20))
# strength = 0.5
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

# Email EU, 500+ days in 45 mil. seconds
# fname = "tnet_sources/email-EU/email-Eu-core-temporal-Dept1.txt"
# timestamp_first = False
# period, time_label = 1, 'd'
# start_timestamp, end_timestamp, add_missing = -1, -1, False
# aggregate_time_to = 7*86400 # week
# aggregate_time_to_range = list(range(3600, 10*86400, 3600))
# strength = 1/86400
# min_cluster = 2

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

# start_timestamp, end_timestamp = -1, -1
# separator = ' '
# mname = None




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