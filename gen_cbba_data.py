from problems import *
import torch
import os

BATCH_SIZE = 100
SEED = 231034871112
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
gen_params = [(100,100), (1,1), None, None, (0,101), (0,0), (1,1), 480, (10,31), (0.25,0.5,0.75,1.0), (30,91)]
# S-CVRPTW Data (more tw)
for n in range(10,51,5):
        m = n//5
    # for m in range(4,5,1):
        out_dir = "data_sample100_stw/s_cvrptw_n{}m{}".format(n,m)
        os.makedirs(out_dir, exist_ok = True)
        data = VRPTW_Dataset.generate(BATCH_SIZE, n, m, *gen_params)
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