#!/usr/bin/env python3

from marpdan.problems import SDVRPTW_Dataset, SDVRPTW_Environment
from marpdan import AttentionLearner
from marpdan.layers import reinforce_loss

import torch

data = SDVRPTW_Dataset.generate(16, 10, 2, 100, min_cust_count = 5)
dyna = SDVRPTW_Environment(data)

learner = AttentionLearner(dyna.CUST_FEAT_SIZE, dyna.VEH_STATE_SIZE, 128)

dyna.reset()

actions = []
logps = []
rewards = []

while not dyna.done:
    if dyna.new_customers:
        learner._encode_customers(dyna.nodes, dyna.cust_mask)
    cust_idx, logp = learner.step(dyna)
    actions.append( (dyna.cur_veh_idx, cust_idx) )
    logps.append( logp )
    r = dyna.step(cust_idx)
    rewards.append( r )

print("Forward ok")
print("Cumul rewards:", torch.stack(rewards).sum(0).squeeze(1))

loss = reinforce_loss(logps, rewards)
loss.backward()

print("Backward ok")

grad = 0
for n,p in learner.named_parameters():
    pnorm = p.pow(2).sum()
    print("{: <64} grad norm: {:.3g}".format(n, pnorm.pow(0.5).item()))
    grad += pnorm
grad.pow_(0.5)
print('-'*96)
print("grad norm: {:.3g}".format(grad.item()))
