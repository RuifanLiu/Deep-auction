from marpdan.problems import *
from marpdan.externals import lkh_solve, ort_solve
from marpdan.utils import eval_apriori_routes

import torch
from torch.utils.data import DataLoader
import pickle
import os

BATCH_SIZE = 10000
SEED = 231034871114
LATE_PS = [0.05, 0.1, 0.2, 0.3, 0.5]
ROLLOUTS = 100

torch.manual_seed(SEED)


# CVRP Data
for n,m in ((10,2), (20,4), (50,10)):
    out_dir = "cvrp_n{}m{}".format(n, m)
    os.makedirs(out_dir, exist_ok = True)

    data = VRPTW_Dataset.generate(BATCH_SIZE, n, m, tw_ratio = 0.0, cust_dur_range = (0,1))

    x_scl = data.nodes[:,:,:2].max() - data.nodes[:,:,:2].min()
    with open(os.path.join(out_dir, "kool_data.pkl"), 'wb') as f:
        pickle.dump(list(zip(
            data.nodes[:,0,:2].div(x_scl).tolist(),
            data.nodes[:,1:,:2].div(x_scl).tolist(),
            data.nodes[:,1:,2].tolist(),
            [data.veh_capa for b in range(BATCH_SIZE)]
            )), f, pickle.HIGHEST_PROTOCOL)

    data.normalize()
    torch.save(data, os.path.join("data", out_dir, "norm_data.pyth"))

# CVRPTW Data
for n,m in ((10,2), (20,4), (50,10)):
    out_dir = "data/cvrptw_n{}m{}".format(n,m)
    os.makedirs(out_dir, exist_ok = True)

    data = VRPTW_Dataset.generate(BATCH_SIZE, n, m)

    data.normalize()
    torch.save(data, os.path.join(out_dir, "norm_data.pyth"))


# S-CVRPTW Data (more tw)
for n,m in ((10,2), (20,4), (50,10)):
    out_dir = "data/s_cvrptw_n{}m{}".format(n,m)
    os.makedirs(out_dir, exist_ok = True)

    data = VRPTW_Dataset.generate(BATCH_SIZE, n, m, tw_ratio = [0.7,0.8,1.0])

    data.normalize()
    torch.save(data, os.path.join(out_dir, "norm_data.pyth"))


# SD-CVRPTW Data
for n,m in ((10,2), (20,4), (50,10)):
    out_dir = "data/sd_cvrptw_n{}m{}".format(n,m)
    os.makedirs(out_dir, exist_ok = True)

    data = SDVRPTW_Dataset.generate(BATCH_SIZE, n, m)
    ort_routes = ort_solve(data)

    data.normalize()
    env = VRPTW_Environment(data)
    ort_costs = eval_apriori_routes(env, ort_routes, 1)

    torch.save(data, os.path.join(out_dir, "norm_data.pyth"))
    torch.save({
        "costs": ort_costs,
        "routes": ort_routes,
        }, os.path.join(out_dir, "ort.pyth"))
