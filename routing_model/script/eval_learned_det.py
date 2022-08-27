from marpdan import AttentionLearner
from marpdan.problems import VRP_Dataset, VRPTW_Dataset, VRPTW_Environment
from marpdan.utils import load_old_weights
from marpdan.dep import tqdm

import torch
from torch.utils.data import DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROLLOUTS = 100

for pb in ("cvrp", "cvrptw"):
    for n in (10, 20, 50):
        m = n // 5
        out_dir = "./results/{}_n{}m{}/".format(pb, n, m)
        data_path = "./data/{}_n{}m{}/norm_data.pyth".format(pb, n, m)
        model_path = "./pretrained/cvrptw_n{}m{}.pyth".format(n, m)

        print(" {}{} ".format(pb, n).center(96, '-'))

        data = torch.load(data_path)
        loader = DataLoader(data, batch_size = 512)

        learner = AttentionLearner(6,4)
        chkpt = torch.load(model_path, map_location = "cpu")
        load_old_weights(learner, chkpt["model"])
        learner.to(dev)
        learner.eval()

        with torch.no_grad():
            # GREEDY
            learner.greedy = True
            costs = []
            logps = []
            for batch in tqdm(loader):
                batch = batch.to(dev)
                env = VRPTW_Environment(data, batch)

                _, logp, rewards = learner(env)
                costs.append( -torch.stack(rewards).sum(0).squeeze(1) )
                logps.append( torch.stack(logp).sum(0).squeeze(1) )
            costs = torch.cat(costs, 0)
            probs = torch.cat(logps, 0).exp()
            print("greedy {:.3f} +- {:.3f} w.p. {:.3g}".format(costs.mean(), costs.std(), probs.mean()))
            torch.save(costs, out_dir + "mardan_greedy.pyth")

            # SAMPLING
            learner.greedy = False
            loader = DataLoader(data, batch_size = 512)
            costs = []
            for batch in tqdm(loader):
                batch = batch.to(dev)
                env = VRPTW_Environment(data, batch)

                roll_costs = []
                roll_logps = []
                for _ in range(ROLLOUTS):
                    _, logp, rewards = learner(env)
                    roll_costs.append( -torch.stack(rewards).sum(0).squeeze(1) )
                    roll_logps.append( torch.stack(logp).sum(0).squeeze(1) )
                best_cost, best_idx = torch.stack(roll_costs).min(0, keepdim=True)
                costs.append( best_cost.squeeze(0) )
                logps.append( torch.stack(roll_logps).gather(0, best_idx).squeeze(0) )
            costs = torch.cat(costs, 0)
            probs = torch.cat(logps, 0).exp()
            print("sample {:.3f} +- {:.3f} w.p. {:.3g}".format(costs.mean(), costs.std(), probs.mean()))
            torch.save(costs, out_dir + "mardan_sample{}.pyth".format(ROLLOUTS))
