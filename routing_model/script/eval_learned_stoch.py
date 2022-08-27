from marpdan import AttentionLearner
from marpdan.problems import SVRPTW_Environment
from marpdan.dep import tqdm
from marpdan.utils import load_old_weights

import torch
from torch.utils.data import DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROLLOUTS = 100
LATE_PS = [0.05, 0.1, 0.2, 0.3, 0.5]

with torch.no_grad():
    for n in (10, 20, 50):
        m = n // 5
        out_dir = "./results/s_cvrptw_n{}m{}/".format(n, m)
        data_path = "./data/s_cvrptw_n{}m{}/norm_data.pyth".format(n, m)
        model_path = "./pretrained/cvrptw_n{}m{}.pyth".format(n, m)

        data = torch.load(data_path)
        loader = DataLoader(data, batch_size = 512)

        learner = AttentionLearner(6,4)
        chkpt = torch.load(model_path, map_location = "cpu")
        load_old_weights(learner, chkpt["model"])
        learner.to(dev)
        learner.eval()
        learner.greedy = True

        costs = []
        for batch in tqdm(loader):
            batch = batch.to(dev)
            env = VRPTW_Environment(data, batch)
            _, _, rewards = learner(env)
            costs.append( -torch.stack(rewards).sum(0).squeeze(1) )
        costs = torch.cat(costs, 0)
        print("latep = 0 : {:.5f} +- {:.5f}".format(costs.mean(), costs.std()))
        torch.save(costs, out_dir + "mardan_late00.pyth")

        for late_p in LATE_PS:
            costs = []
            for batch in tqdm(loader):
                batch = batch.to(dev)
                env = SVRPTW_Environment(data, batch, late_p = late_p)

                roll_costs = []
                for _ in range(ROLLOUTS):
                    _,_, rewards = learner(env)
                    roll_costs.append( -torch.stack(rewards).sum(0).squeeze(1) )
                costs.append( torch.stack(roll_costs).mean(0) )
            costs = torch.cat(costs, 0)
            print("latep = {} : {:.5f} +- {:.5f}".format(late_p, costs.mean(), costs.std()))
            torch.save(costs, out_dir + "mardan_late{:02.0f}.pyth".format(100*late_p))

