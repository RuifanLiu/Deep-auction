#!/usr/bin/env python3

from marpdan import *
from marpdan.dep import *
from marpdan.utils import *

import torch

pyplot.ion()

torch.manual_seed(546789)

#### BASE DATA
data = VRPTWDataset.generate(1, None, 10, 2)
ort_routes, _, _ = call_ortools(data)
data.normalize()
dyna = VRPTWDynamics(data)

#### MODEL
chkpt = torch.load("output/chkpt_ep34.pyth", map_location = "cpu")
model = AttentionLearner(data.CUST_FEAT_SIZE, dyna.VEH_STATE_SIZE)
model.load_state_dict(chkpt["model"])
model.eval()

#### REF PLOT
fig,(ax1,ax2, ax3) = pyplot.subplots(1,3)

ax1.set_title("ORTools")
plot_customers(ax1, data.customers[0])
plot_routes(ax1, data.customers[0], ort_routes[0])

ax2.set_title("Learned")
actions, _, _ = model(dyna)
plot_customers(ax2, data.customers[0])
plot_actions(ax2, data.customers[0], [(i.item(), j.item()) for i,j in actions])

ax3.set_title("Online")

### DYNA CUST
dod = 0.5
hid_count = int(dod * 10)
hidden_idx = (1+torch.randperm(10))[:hid_count].unsqueeze(0)
hidden = torch.zeros(1, 11, dtype = torch.uint8).scatter_(1, hidden_idx, 1)
reveal = torch.zeros(1,11).scatter_(1, hidden_idx, 0.5*torch.rand(1,11)+0.05)
print("Reveal times =", reveal[0].tolist())

plot_customers(ax3, data.customers[0])
patches = {}
for j in hidden_idx[0].tolist():
    patches[j] = ax3.scatter(*zip(data.customers[0,j,:2].tolist()), 200, 'w', 'o', zorder = 10)

### INIT TRAJ
dyna.reset()
dyna.mask = dyna.mask | hidden.unsqueeze(1).expand(-1,2,-1)
dyna.cur_veh_mask = dyna.cur_veh_mask | hidden
done = False
model._encode_customers(dyna.customers, hidden)

cost = 0
pxs,pys = data.customers[0, 0, :2].unsqueeze(1).expand(-1, 2).tolist()

# TRAJ
while not done:
    i = dyna.cur_veh_idx.item()
    j, logp = model.step(dyna)
    print("At t = {:.3f}, {} goes to {}".format( dyna.cur_veh[...,3].item(), i, j.item() ))
    input()
    r, done = dyna.step(j)
    cost -= r.item()

    to_reveal = (dyna.cur_veh[:,:,3] > reveal) & hidden
    if to_reveal.any():
        print("At t = {:.3f}: revealing {}".format(
            dyna.cur_veh[...,3].item(), to_reveal.nonzero()[:,1].tolist()))
        for rev_j in to_reveal.nonzero()[:,1].tolist():
            patches[rev_j].remove()
        hidden -= to_reveal
        dyna.mask -= to_reveal.unsqueeze(1).expand(-1,2,-1)
        dyna.cur_veh_mask -= to_reveal
        model._encode_customers(dyna.customers, hidden)
    
    x,y = data.customers[0,j.item(),:2].tolist()
    ls = '--' if j == 0 else '-'
    ax3.plot([pxs[i], x], [pys[i], y], color = (0, i, 1-i), zorder = -1, linestyle = ls)
    pxs[i], pys[i] = x,y
    pyplot.pause(0.1)

print("Done")
input()
