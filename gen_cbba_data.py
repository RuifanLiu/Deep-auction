from problems import *
import torch
import os

BATCH_SIZE = 10
SEED = 231034871114
LATE_PS = [0.05, 0.1, 0.2, 0.3, 0.5]
ROLLOUTS = 100

torch.manual_seed(SEED)

'''
# CVRP Data
for n,m in ((10,2), (20,4), (50,10)):
    out_dir = "cvrp_n{}m{}".format(n, m)
    os.makedirs(out_dir, exist_ok = True)

    data = VRPTW_Dataset.generate(BATCH_SIZE, n, m, tw_ratio = 0.0, cust_dur_range = (0,1))

    data.normalize()
    torch.save(data, os.path.join("data", out_dir, "norm_data.pyth"))

# CVRPTW Data
for n,m in ((10,2), (20,4), (50,10)):
    out_dir = "data/cvrptw_n{}m{}".format(n,m)
    os.makedirs(out_dir, exist_ok = True)

    data = VRPTW_Dataset.generate(BATCH_SIZE, n, m)

    data.normalize()
    torch.save(data, os.path.join(out_dir, "norm_data.pyth"))
'''

# S-CVRPTW Data (more tw)
for n in range(10,101,10):
    for m in range(2,11,1):
        out_dir = "data_test/s_cvrptw_n{}m{}".format(n,m)
        os.makedirs(out_dir, exist_ok = True)
        data = VRPTW_Dataset.generate(BATCH_SIZE, n, m, tw_ratio = 1.0)
        data.normalize()
        torch.save(data, os.path.join(out_dir, "norm_data.pyth"))
'''
# SD-CVRPTW Data
for n in range(10,101,10):
    for m in range(2,11,1):

        out_dir = "data/sd_cvrptw_n{}m{}".format(n,m)
        os.makedirs(out_dir, exist_ok = True)

        data = SDVRPTW_Dataset.generate(BATCH_SIZE, n, m, tw_ratio = 1.0)
        data.normalize()
        torch.save(data, os.path.join(out_dir, "norm_data.pyth"))
'''