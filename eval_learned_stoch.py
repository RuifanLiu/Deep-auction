from routing_model._learner import AttentionLearner
from problems import SVRPTW_Environment, VRPTW_Environment, VRPTW_Dataset
from routing_model.dep import tqdm
from routing_model.utils import load_old_weights
from routing_model.baselines import *

import torch
from torch.utils.data import DataLoader
from cbba._data_util import PaddedData

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROLLOUTS = 100
LATE_PS = [0.05, 0.1, 0.2, 0.3, 0.5]

with torch.no_grad():
    for n in [100]:
        m = 1
        out_dir = "./results/s_cvrptw_n{}m{}/".format(n, m)
        # data_path = "./train_data/s_cvrptw_n1-{}m{}/norm_data.pyth".format(n, m)
        model_path = 'vrp_output/SVRPTWn100m1_220902-1529/chkpt_ep100.pyth'

        gen_params = [10, 1, (100,100), (1, 3), 0, (0, 101), (0, 0), (1, 1),\
            480, (10, 31), 1.0, (30, 91)]
        data = VRPTW_Dataset.generate(1000, *gen_params)
        data.normalize()

        # data = torch.load(data_path)
        loader = DataLoader(data, batch_size = 1)

        learner = AttentionLearner(7,5)
        chkpt = torch.load(model_path, map_location=torch.device('cpu'))
        learner.load_state_dict(chkpt["model"])
        learner.to(dev)
        learner.eval()
        learner.greedy = True

        bl_wrapped_learner =  CriticBaseline(learner, cust_count = 100, use_qval=False, use_cumul_reward=True)
        bl_wrapped_learner.load_state_dict(chkpt['critic'])
        bl_wrapped_learner.to(dev)

        rews = []
        for batch in tqdm(loader):
            # batch = batch.to(dev)
            if data.cust_mask is None:
                vehs, custs, mask = batch[0].to(dev), batch[1].to(dev), None
            else:
                vehs, custs, mask = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)

            padded_data = PaddedData(vehs=vehs, nodes=custs, padding_size=bl_wrapped_learner.cust_count+1)
            env = SVRPTW_Environment(padded_data, None, None, None, *[2, 0.9, 0.1, 0.05, 0.5, 0.2])
            # env = SVRPTW_Environment(batch, vehs, custs, mask, *[2, 0.9, 0.1, 0.05, 0.5, 0.2])
            actions, logps, rewards, bl_vals = bl_wrapped_learner(env)
            print(f"actions from baseline = {actions}")
            print(f"rewards from baseline = {rewards}")
            # _, _, rewards = learner(env)
            # rews.append( torch.stack(rewards).sum(0).squeeze(1).mean())
            rews.append( rewards.mean())
        rews = torch.stack(rews)
        print("latep = 0 : {:.5f} +- {:.5f}".format(rews.mean(), rews.std()))

        # for late_p in LATE_PS:
        #     costs = []
        #     for batch in tqdm(loader):
        #         # batch = batch.to(dev)
        #         env = SVRPTW_Environment(data, batch, late_p = late_p)

        #         roll_costs = []
        #         for _ in range(ROLLOUTS):
        #             _,_, rewards = learner(env)
        #             roll_costs.append( -torch.stack(rewards).sum(0).squeeze(1) )
        #         costs.append( torch.stack(roll_costs).mean(0) )
        #     costs = torch.cat(costs, 0)
        #     print("latep = {} : {:.5f} +- {:.5f}".format(late_p, costs.mean(), costs.std()))
        #     torch.save(costs, out_dir + "mardan_late{:02.0f}.pyth".format(100*late_p))

