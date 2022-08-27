from marpdan import AttentionLearner
from marpdan.problems import VRPTW_Dataset, VRPTW_Environment
from marpdan.externals import call_ortools
from marpdan.utils import *
from marpdan.dep import matplotlib as mpl, pyplot as plt

import torch

SEED = 12348877555
BATCH_SIZE = 128

mpl.rcParams["axes.titlesize"] = 20

chkpt = torch.load("./output/VRPTWn20m4_190314-1011/chkpt_ep20.pyth", map_location = "cpu")
learner = AttentionLearner(6,4)
learner.load_state_dict(chkpt["model"])
learner.eval()

torch.manual_seed(SEED)

data = VRPTW_Dataset.generate(BATCH_SIZE,20,4)

ref_routes, costs, pens = call_ortools(data)
x_scl, t_scl = data.normalize()
ref_costs = torch.tensor([c/x_scl + p/t_scl for c,p in zip(costs, pens)])

env = VRPTW_Environment(data)
with torch.no_grad():
    actions, logps, rewards = learner(env)

costs = torch.stack(rewards).sum(0).mul(-1).squeeze(1)

gaps = costs / ref_costs - 1
gaps, sub_idx = gaps.sort()

fill_ratio = 1-env.vehicles[:,:,2]
print("Mean fill ratio on deployed vehicles: {:.0%}".format(fill_ratio[fill_ratio > 0].mean()))

print("      {: ^5} {: ^5} {: ^5} {: ^5} {: ^5}".format("|-", "[", "|", "]", "-|"))
print("Gaps: {:5.0%} {:5.0%} {:5.0%} {:5.0%} {:5.0%}".format(
    gaps[0], gaps[BATCH_SIZE//4], gaps[BATCH_SIZE//2], gaps[3*BATCH_SIZE//4], gaps[-1]))

sub_idx = torch.cat((sub_idx[:4], sub_idx[BATCH_SIZE//2-4:BATCH_SIZE//2], sub_idx[-4:]))

i = 0
for cust,acts,rs,c,ref in zip(data.nodes[sub_idx],
        ([(i[n].item(), j[n].item()) for (i,j) in actions] for n in sub_idx),
        (ref_routes[n] for n in sub_idx),
        costs[sub_idx],ref_costs[sub_idx]):
    fig,(ref_ax,ax) = plt.subplots(1,2)

    g = c / ref - 1
    ref_ax.set_title("ORTools (cost = {:.3f})".format(ref))
    ax.set_title("Learned (cost = {:.3f}, gap = {:.0%})".format(c,g))

    plot_customers(ref_ax, cust)
    plot_routes(ref_ax,cust,rs)
    plot_customers(ax, cust)
    plot_actions(ax,cust,acts,4)

    fig.set_tight_layout(True)
    fig.set_size_inches(16,9)

    fig.savefig("results/routes_n20m4_{:02}_{:.0f}.pdf".format(i,100*g))
    i += 1

plt.show()
