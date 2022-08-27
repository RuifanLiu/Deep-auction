from marpdan.problems import VRPTW_Environment, SVRPTW_Environment
from marpdan.dep import tqdm

import torch
from torch.utils.data import DataLoader

ROLLOUTS = 100
LATE_PS = [0.05, 0.1, 0.2, 0.3, 0.5]

for n in (10, 20, 50):
    m = n // 5
    out_dir = "./results/s_cvrptw_n{}m{}/".format(n, m)
    data_path = "./data/s_cvrptw_n{}m{}/norm_data.pyth".format(n, m)

    data = torch.load(data_path)
    loader = DataLoader(data, batch_size = 512)

    nodes = data.nodes.clone()
    nodes[:,:,:2] *= 100
    nodes[:,:,2] *= 200
    nodes[:,:,3:] *= 480
    unnormed = VRPTW_Dataset(data.veh_count, 200, 1, nodes)
    ort_optim_routes = ort_solve(unnormed)

    ort_costs = eval_apriori_routes(VRPTW_Environment(data), ort_optim_routes, 1)
    torch.save({"costs": ort_costs, "routes": ort_routes}, out_dir + "ort_optim_late00.pyth")

    for late_p in LATE_PS:
        optim_costs  = []
        expect_costs = []

        unnormed.veh_speed = 1 - 0.5*late_p
        ort_expect_routes = ort_solve(unnormed)

        for batch in tqdm(loader):
            env = SVRPTW_Environment(data, batch, late_p = late_p)
            roll_costs = []
            for _ in range(ROLLOUTS):
                roll_costs.append( eval_apriori_routes(env, ort_optim_routes, 1) )
            optim_costs.append( torch.stack(roll_costs).mean(0) )
            roll_costs = []
            for _ in range(ROLLOUTS):
                roll_costs.append( eval_apriori_routes(env, ort_expect_routes, 1) )
            expect_costs.append( torch.stack(roll_costs).mean(0) )

        optim_costs = torch.cat(optim_costs, 0)
        expect_costs = torch.cat(expect_costs, 0)
        print("latep = {} (optim): {:.5f} +- {:.5f}".format(late_p, optim_costs.mean(), optim_costs.std()))
        print("latep = {} (expect): {:.5f} +- {:.5f}".format(late_p, expect_costs.mean(), expect_costs.std()))

        torch.save({"costs": optim_costs, "routes": None},
                out_dir + "ort_optim_late{:02.0f}.pyth".format(100*late_p))
        torch.save({"costs":expect_costs, "routes": ort_expect_routes},
                out_dir + "ort_expect_late{:02.0f}.pyth".format(100*late_p))
