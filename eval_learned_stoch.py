from routing_model._learner import AttentionLearner
from problems import SVRPTW_Environment, VRPTW_Environment, VRPTW_Dataset
from routing_model.dep import tqdm
from routing_model.utils import load_old_weights
from routing_model.baselines import *

import os
import torch
from torch.utils.data import DataLoader
from cbba._data_util import PaddedData
from cbba._eval import eval_routes_drl

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROLLOUTS = 100
LATE_PS = [0.05, 0.1, 0.2, 0.3, 0.5]
SPEED_VARS = [0.0, 0.05, 0.1, 0.2]
# SPEED_VARS = [0.0]


with torch.no_grad():
    for n in [100]:
        m = n//5
        out_dir = "./results/s_cvrptw_n{}m{}/".format(n, m)
        os.makedirs(out_dir, exist_ok = True)
        fpath = os.path.join(out_dir, "marl_result.csv")
        data_path = "./data_sample10_stw/s_cvrptw_n{}m{}/norm_data.pyth".format(n, m)
        # model_path = 'vrp_output/SVRPTWn10m2_240430-1031/chkpt_ep100.pyth'
        if n==10: model_path = 'vrp_output/SVRPTWn10m2_221106-1646/chkpt_ep100.pyth'
        elif n==20: model_path = 'vrp_output/SVRPTWn20m4_221106-1649/chkpt_ep100.pyth'
        elif n==50: model_path = 'vrp_output/SVRPTWn50m10_221106-1555/chkpt_ep50.pyth'
        elif n==100: model_path = 'vrp_output/SVRPTWn50m10_221106-1555/chkpt_ep50.pyth'

        # gen_params = [(100,100), (1,1), None, None, (0,101), (0,0), (1,1), \
        #     480, (10,31), (0.25,0.5,0.75,1.0), (30,91)]
        # data = VRPTW_Dataset.generate(ROLLOUTS, n, m, *gen_params)
        # data.normalize()
        data = torch.load(data_path)
        loader = DataLoader(data, batch_size = 1)

        learner = AttentionLearner(7,5) 
        chkpt = torch.load(model_path, map_location=torch.device('cpu'))
        learner.load_state_dict(chkpt["model"])
        learner.to(dev)
        learner.eval()
        learner.greedy = True

        # bl_wrapped_learner =  CriticBaseline(learner, cust_count = n, use_qval=False, use_cumul_reward=True)
        # bl_wrapped_learner.load_state_dict(chkpt['critic'])
        # bl_wrapped_learner.to(dev)

        for speed_var in SPEED_VARS:
            rews = []
            for batch in tqdm(loader):
                # batch = batch.to(dev)
                if data.cust_mask is None:
                    vehs, custs, mask = batch[0].to(dev), batch[1].to(dev), None
                else:
                    vehs, custs, mask = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)

                padded_data = PaddedData(vehs=vehs, nodes=custs, cust_mask=mask)
                env = SVRPTW_Environment(padded_data, None, None, None, *[1, 4, speed_var, 0.0, 0.5, 0.2])
                roll_rews = []
                for _ in range(ROLLOUTS):
                    # actions, logps, rewards, bl_vals = bl_wrapped_learner(env, greedy=True)
                    actions, logps, rewards = learner(env)
                
                # print(f"actions from baseline = {actions}")
                # print(f"rewards from baseline = {rewards}")
                    roll_rews.append(torch.stack(rewards).sum(0))
                rews.append( torch.stack(roll_rews).mean(0))
            rews = torch.cat(rews, 0)
            print("speed variance = {} : {:.5f} +- {:.5f}".format(speed_var, rews.mean(), rews.std()))
            # torch.save(rews, out_dir+'speed_var_{:02.0f}.pyth'.format(100*speed_var))

            with open(fpath, 'a') as f:
                f.write("Task number: {} Vehicle number: {} Speed var: {} Rewards mean: {} Reward std:{} \n".format(n, m, speed_var, rews.mean(), rews.std()))

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

