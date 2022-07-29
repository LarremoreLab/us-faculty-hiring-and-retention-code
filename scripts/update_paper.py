import usfhn.paper_stats
import usfhn.paper_tables
import usfhn.plots


if __name__ == '__main__':
    usfhn.paper_stats.write_stats()
    usfhn.paper_tables.write_tables()
    usfhn.plots.plot_to_paper_figures()
