#!/usr/bin/env python3
from marpdan.problems import VRP_Dataset, VRPTW_Dataset, SDVRPTW_Dataset
from marpdan.dep import MPL_ENABLED, pyplot
from marpdan.utils import plot_customers, setup_axes_layout

data = VRP_Dataset.generate(3, 8, 2, min_cust_count = 5)

data_tw = VRPTW_Dataset.generate(3, 8, 2, min_cust_count = 5)

data_sdtw = SDVRPTW_Dataset.generate(12,8,2, dod = (0.25,0.5,0.75), d_early_ratio = (0,0.5,1.0), tw_ratio = (0.25,0.5,0.75,1.0) )

if MPL_ENABLED:
    fig = pyplot.figure()
    axes = setup_axes_layout(fig, 6)
    for cust, m, ax in zip(data.nodes, data.cust_mask, axes[0]):
        plot_customers(ax, cust[m ^ 1], detailed = True)

    for cust, m, ax in zip(data_tw.nodes, data_tw.cust_mask, axes[1]):
        plot_customers(ax, cust[m ^ 1], detailed = True)

    fig = pyplot.figure()
    axes = setup_axes_layout(fig, 12)
    for cust, ax in zip(data_sdtw.nodes, axes.flatten()):
        plot_customers(ax, cust)

    pyplot.show()
else:
    print(data.nodes)
    print(data_tw.nodes)
