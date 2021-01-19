import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    from importer import DataContainer, LYONSCHOOL_SETTINGS_BASE, LYONSCHOOL_EVALUATE_RANGE
    evaluate_at = list(LYONSCHOOL_EVALUATE_RANGE)
    dc = DataContainer(LYONSCHOOL_SETTINGS_BASE)

    tgraphs = []
    sgraphs = []
    ylabel = ""


    for aggregate_time_to in evaluate_at:
        tg = dc.aggregate_to(aggregate_time_to, makecopy=True).get_timegraph()
        tgraphs.append(tg)

        # Uncomment the following lines when plotting layout statistics (currently none yet)
        # sg = SugiyamaLayout(tg)
        # sg.set_locations()
        # sgraphs.append(sg)

    xlabel = "Aggregation"

    def plot(ylab, res):
        plt.figure()
        plt.plot(evaluate_at, res)
        plt.scatter(evaluate_at, res)
        plt.xlabel(xlabel)
        plt.ylabel(ylab)

    ylabel = "number of clusters"
    results = list(map(lambda tg: tg.num_clusters(), tgraphs))
    plot(ylabel, results)

    ylabel = "Average relative continuity"
    results = list(map(lambda tg: tg.average_relative_continuity(), tgraphs))
    plot(ylabel, results)

    ylabel = "Average absolute continuity"
    results = list(map(lambda tg: tg.average_absolute_continuity(), tgraphs))
    plot(ylabel, results)

    ylabel = "Average relative continuity difference"
    results = list(map(lambda tg: tg.average_relative_continuity_diff(), tgraphs))
    plot(ylabel, results)

    ylabel = "Average absolute continuity difference"
    results = list(map(lambda tg: tg.average_absolute_continuity_diff(), tgraphs))
    plot(ylabel, results)

    ylabel = "Normalized absolute continuity"
    results = list(map(lambda tg: tg.normalized_absolute_continuity(), tgraphs))
    plot(ylabel, results)

    ylabel = "Normalized absolute continuity difference"
    results = list(map(lambda tg: tg.normalized_absolute_continuity_diff(), tgraphs))
    plot(ylabel, results)

    ylabel = "Variance of absolute continuity difference"
    results = list(map(lambda tg: np.var(tg.absolute_continuity_diff()), tgraphs))
    plot(ylabel, results)

    plt.show()
