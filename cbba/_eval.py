import torch
import numpy as np
from routing_model.utils._misc import _pad_with_zeros
from cbba._data_util import PaddedData

def eval_routes_drl(Environment, batch, drl_model, value_model, assignments):
    vehs, custs = batch
    batch_size, nodes_count, node_state_size  = custs.size()
    _, n_agent, _ = vehs.size()
    iterations = 1

    rewards = 0
    for n_iter in range(iterations):
        for n in range(n_agent):
            task_list_end = np.argwhere(assignments[n]==-1)
            task_list = assignments[n][:task_list_end[0][0]] if len(task_list_end) else assignments[n]
            # unmask selected tasks and depot
            trunct_custs = custs[:, [0]+list(task_list), :]
            
            # custs.new_zeros((batch_size, value_model.cust_count+1, node_state_size), dtype = torch.float)\
            #     .scatter(1,torch.tensor([0]+list(task_list))[None,:,None].expand(-1,-1,node_state_size),custs)
            # mask = custs.new_ones((batch_size, value_model.cust_count+1), dtype = torch.bool)\
            #     .scatter(1,torch.tensor([0]+list(task_list))[None,:],0)
            # data = PaddedData(vehs=vehs[:,n,:][:,None,:], nodes=trunct_custs, cust_mask= mask)
            data = PaddedData(vehs=vehs[:,n,:][:,None,:], nodes=trunct_custs, padding_size=value_model.cust_count+1)
            dyna = Environment(data, None, None, None, *[2, 0.9, 0.1, 0.05, 0.5, 0.2])
            acts, logps, rew = drl_model(dyna)
            acts1, logps1, rew1, bls = value_model(dyna) 

            # print(f"tasklist = {task_list}")
            # print(f"actions from learner = {acts}")
            # print(f"actions from baseline = {acts1}")
            # print(f"rewards from learner = {rew}")
            # print(f"rewards from baseline = {rew1}")
            # print(f"baseline value from baseline = {bls}")

            # rewards += torch.stack(rew).sum(0).squeeze(1)
            rewards += rew1.mean()
    
    return rewards/iterations, None

def eval_apriori_routes(Environment, batch, assignments):
    vehs, custs = batch
    batch_size, nodes_count, d  = custs.size()
    _, n_agent, _ = vehs.size()

    for n in range(n_agent):
        task_list_end = np.argwhere(assignments[n]==-1)
        assignments[n] = assignments[n][:task_list_end[0][0]] if len(task_list_end) else assignments[n]
    
    task_list = torch.tensor([0] + list(np.concatenate(assignments)))
    mask = custs.new_ones((batch_size, nodes_count), dtype = torch.bool)\
                .scatter(1,torch.tensor([0]+list(task_list))[None,:],0)
    data = PaddedData(vehs=vehs, nodes=custs, cust_mask= mask)
    dyna = Environment(data)
    
    rewards = 0
    iterations = 1
    for n_iter in range(iterations):
        dyna.reset()
        routes_it = [_pad_with_zeros(inst_routes) for inst_routes in assignments]
        rew = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor([[next(routes_it[dyna.cur_veh_idx.item()])]], dtype = torch.int64)
            rew.append( dyna.step(cust_idx) )
        rewards += torch.stack(rew).sum(dim = 0).squeeze(-1)
    
    return rewards/iterations, None
