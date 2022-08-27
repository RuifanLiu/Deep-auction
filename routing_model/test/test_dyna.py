#!/usr/bin/env python3
from marpdan.problems import VRP_Dataset, VRP_Environment
from marpdan.dep import MPL_ENABLED, pyplot
from marpdan.utils import plot_customers
import torch

if not MPL_ENABLED:
    raise RuntimeError("Cannot run this test without matplotlib")

pyplot.ion()

data = VRP_Dataset.generate(1, 10, 2, 100)

fig,ax = pyplot.subplots()
plot_customers(ax, data.nodes[0])

dyna = VRP_Environment(data)

print("CVRP with {} vehicle{} of capacity {} and {} customers".format(
    dyna.veh_count, 
    "s" if dyna.veh_count > 1 else "",
    dyna.veh_capa,
    dyna.nodes_count-1)
    )
print("{: ^16} {: ^16} {: ^16} {: ^16}".format("CUST #", "X", "Y", "DEM"))
for j,(x,y,d) in enumerate(dyna.nodes[0].tolist()):
    print("{: ^16} {: ^16.2f} {: ^16.2f} {: ^16.2f}".format(j,x,y,d))

dyna.reset()
val = 0
while not dyna.done:
    vs = dyna.cur_veh[0,0].tolist()
    i = dyna.cur_veh_idx.item()
    print("Choose action for vehicle #{}: {}".format(i, vs))
    avail = (dyna.cur_veh_mask[0,0] ^ 1).nonzero().squeeze(1).tolist()
    j = -1
    while j not in avail:
        try:
            j = int( input("{} > ".format(avail)) )
        except ValueError:
            continue
    r = dyna.step( torch.tensor([[j]]) )
    vss = dyna.vehicles[0,i].tolist()
    ax.plot([vs[0], vss[0]],[vs[1], vss[1]], color = (0, 1 - i, i), zorder = -1)
    pyplot.pause(0.01)
    print("Received {:.3f} of reward".format(r.item()))
    val += r.item()
print("Total reward = {:.3f}".format(val))
