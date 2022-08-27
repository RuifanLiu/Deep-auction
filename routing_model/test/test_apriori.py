#!/usr/bin/env python3

import torch

from marpdan.problems import VRPTW_Dataset, SVRPTW_Environment
from marpdan.externals import call_ortools
from marpdan.utils import eval_apriori_routes
from marpdan.dep import pyplot as plt, matplotlib as mpl

data = VRPTW_Dataset.generate(128, 10, 2)
ref_routes, costs, pens = call_ortools(data)
ref_costs = [c + p for c,p in zip(costs, pens)]

gaps = []
for late_p in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    dyna = SVRPTW_Environment(data, late_p = late_p)
    deg_costs = eval_apriori_routes(dyna, ref_routes, 100)

    print("With late proba = {:.3f}, degradation of a-priori solution due to stochastic travel time = {:.0%}".format(
        late_p,
        sum(deg_c / c - 1 for deg_c, c in zip(deg_costs, ref_costs)) / len(deg_costs)
        ))
