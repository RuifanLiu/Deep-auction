import torch
import numpy as np
from routing_model.utils._misc import _pad_with_zeros
from cbba._data_util import PaddedData

def eval_routes_drl(Environment, batch, drl_model, assignments):
    vehs, custs = batch
    n_agent = len(vehs)
    iterations = 100

    rewards = 0
    for n_iter in range(iterations):
        for n in range(n_agent):
            task_list_end = np.argwhere(assignments[n]==-1)
            task_list = assignments[n][:task_list_end[0][0]] if len(task_list_end) else assignments[n]
            task_list = [0] + task_list
            custs_assigned = custs[:,task_list]
            veh = vehs[:,n,:][:,None,:]
            data = PaddedData(vehs=vehs, nodes=custs_assigned)

            dyna = Environment(data)
            _, _, rew = drl_model(dyna)
            rewards += torch.stack(rew).sum(0).squeeze(1)
    
    return rewards/iterations, None

def eval_apriori_routes(Environment, batch, assignments):
    vehs, custs = batch
    iterations = 100
    dyna = Environment(batch, vehs, custs)
    rewards = 0
    for n_iter in range(iterations):
        dyna.reset()
        routes_it = [[_pad_with_zeros(route) for route in inst_routes] for inst_routes in assignments]
        rew = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor([[next(routes_it[n][i.item()])]
            for n,i in enumerate(dyna.cur_veh_idx)], dtype = torch.int64)
            rew.append( dyna.step(cust_idx) )
        rewards += torch.stack(rew).sum(dim = 0).squeeze(-1)
    
    return rewards/iterations, None
