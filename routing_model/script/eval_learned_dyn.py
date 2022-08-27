from marpdan import AttentionLearner
from marpdan.problems import SDVRPTW_Environment
from marpdan.dep import tqdm

import torch
from torch.utils.data import DataLoader

for n in (10, 20):#, 50):
    data = torch.load("./data/sd_cvrptw_n{}m{}/norm_data.pyth".format(n, n // 5))
    loader = DataLoader(data, batch_size=50)

    learner =  AttentionLearner(7,4)
    chkpt = torch.load("./pretrained/sd_cvrptw_n{}m{}.pyth".format(n, n // 5), map_location='cpu')
    learner.load_state_dict(chkpt["model"])
    learner.eval()
    learner.greedy = True
    
    costs = []
    qos = []
    with torch.no_grad():
        for b,batch in enumerate(tqdm(loader)):
            env = SDVRPTW_Environment(data, batch, pending_cost=0, late_p=0)
            _,_,rewards = learner(env)
            costs.append( -torch.stack(rewards).sum(0).squeeze(-1) )
            pending = (env.served ^ True).float().sum(-1) - 1
            qos.append(1 - pending / (env.nodes_count - 1))
    
        costs = torch.cat(costs, 0)
        qos = torch.cat(qos, 0)

    dods = (data.nodes[:,:,6] > 0).sum(1).float() / (data.nodes.size(1)-1)
    for k,subset in (("leq40", dods <= 0.4),
            ("less60", (0.4 < dods) & (dods < 0.6)),
            ("geq60", 0.6 <= dods)):
        print("{}: {:5.2f} +- {:5.2f} (qos={:.2%})".format(k, costs[subset].mean(), costs[subset].std(),
                                                           qos[subset].mean()))
        torch.save({"costs":costs[subset], "qos":qos[subset]},
                "./results/sd_cvrptw_n{}m{}/mardan_{}.pyth".format(n, n // 5, k))
