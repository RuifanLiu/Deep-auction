#!/usr/bin/env python3
from marpdan import AttentionLearner
from marpdan.problems import VRP_Dataset, VRP_Environment, VRPTW_Dataset, VRPTW_Environment
from marpdan.layers import reinforce_loss

data = VRP_Dataset.generate(4, 10, 2)
dyna = VRP_Environment(data)

learner = AttentionLearner(data.CUST_FEAT_SIZE, dyna.VEH_STATE_SIZE, tanh_xplor = 10)

actions, logps, rewards = learner(dyna)
print("Forward pass ok for VRP")

loss = reinforce_loss(logps, rewards)
loss.backward()
print("Backward pass ok for VRP")



data = VRPTW_Dataset.generate(4, 10, 2)
dyna = VRPTW_Environment(data)

learner = AttentionLearner(data.CUST_FEAT_SIZE, dyna.VEH_STATE_SIZE, tanh_xplor = 10)

actions, logps, rewards = learner(dyna)
print("Forward pass ok for VRPTW")

loss = reinforce_loss(logps, rewards)
loss.backward()
print("Backward pass ok for VRPTW")
