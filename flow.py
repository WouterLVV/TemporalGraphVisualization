from tgv.layout import SizedConnectionLayout

from tgv.data_importer import DataContainer, LYONSCHOOL_SETTINGS_DEFAULT

if __name__ == '__main__':
    settings = LYONSCHOOL_SETTINGS_DEFAULT
    dc = DataContainer(settings)
    tg = dc.get_timegraph()

    sg = SizedConnectionLayout(tg, line_width=1,
                               cluster_height_method='linear',
                               horizontal_density=1,
                               vertical_density=0.5)
    print(f"Number of crossings in graph: {sg.number_of_crossings()}")
    print(f"Longest chain in graph: {sg.longest_chain_length()}")
    print(f"Average chain length in graph: {sg.average_chain_length(): .2f}")
    sg.draw_graph(filename="flow_output/"+dc.net_name+dc.net_suffix+".svg",
                  colormap=dc.settings.colormap,
                  timestamp_translator=dc.timestamp_reverse_translator,
                  show_annotations=False,
                  show_timestamps=True,
                  #stats_info=("homogeneity_diff", "homogeneity", "in_out_difference", "layer_num_clusters", "layer_num_members"),
                  fading=True,
                  emphasize_communities=True)  # Shows connections between aligned clusters lighter than unaligned clusters
    print(dc.net_name+dc.net_suffix)