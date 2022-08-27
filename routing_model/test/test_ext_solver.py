#!/usr/bin/env python3
from marpdan.externals import ort_solve, lkh_solve
from marpdan.problems import VRP_Dataset, VRPTW_Dataset, VRP_Environment, VRPTW_Environment
from marpdan.dep import *
from marpdan.utils import *


for pbtype, pbenv in [(VRP_Dataset, VRP_Environment), (VRPTW_Dataset, VRPTW_Environment)]:
    for solver in [ort_solve, lkh_solve]:
        data = pbtype.generate(4, 10, 2)
        routes = solver(data)
        data.normalize()
        env = pbenv(data)
        costs = eval_apriori_routes(env, routes, 1)
        for rs, c in zip(routes, costs):
            print("Cost = {}".format(c))
            print(routes_to_string(rs))

        if MPL_ENABLED:
            fig = pyplot.figure()
            axes = setup_axes_layout(fig, 4, 1)
            for cust, m, rs, c, ax in zip(data.nodes, data.cust_mask, routes, costs, axes.flatten()):
                ax.set_title("Cost = {}".format(c))
                plot_customers(ax, cust[m ^ 1])
                plot_routes(ax, cust[m ^ 1], rs)
if MPL_ENABLED:
    pyplot.show()
