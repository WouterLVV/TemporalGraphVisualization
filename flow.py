from tgv.layout import SizedConnectionLayout

from tgv.data_importer import DataContainer, LYONSCHOOL_SETTINGS_DEFAULT

if __name__ == '__main__':

    dc = DataContainer(LYONSCHOOL_SETTINGS_DEFAULT)
    tg = dc.get_timegraph()

    sg = SizedConnectionLayout(tg, line_width=1,
                               cluster_heigsht_method='linear',
                               horizontal_density=1,
                               vertical_density=0.5)

    sg.set_order(barycenter_passes=10)
    sg.set_alignment(stairs_iterations=5)
    sg.draw_graph(filename="flow_output/"+dc.net_name+dc.net_suffix+".svg",
                  colormap=dc.settings.colormap,
                  timestamp_translator=dc.timestamp_reverse_translator,
                  show_annotations=False,
                  show_timestamps=True,
                  stats_info=("homogeneity_diff", "homogeneity", "in_out_difference", "layer_num_clusters", "layer_num_members"),
                  fading=True)
    print(dc.net_name+dc.net_suffix)
