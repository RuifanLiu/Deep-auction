import torch
import numpy as np
from routing_model.utils._misc import _pad_with_zeros
from cbba._data_util import PaddedData
from problems import *

def eval_routes_drl(args, Environment, batch, policy_model, assignments):
    vehs, custs = batch
    batch_size, nodes_count, node_state_size  = custs.size()
    _, n_agent, _ = vehs.size()

    if issubclass(Environment, VRPTW_Environment):
        env_params = [args.pending_cost, args.late_discount]
        if issubclass(Environment, SVRPTW_Environment):
            env_params.extend([args.speed_var, args.late_prob, args.slow_down, args.late_var])

    iterations = 100
    rewards = 0
    for n_iter in range(iterations):
        for n in range(n_agent):
            task_list_end = np.argwhere(assignments[n]==-1)
            task_list = assignments[n][:task_list_end[0][0]] if len(task_list_end) else assignments[n]
            # unmask selected tasks and depot
            trunct_custs = custs[:, [0]+list(task_list), :]
        
            data = PaddedData(vehs=vehs[:,n,:][:,None,:], nodes=trunct_custs, padding_size=policy_model.cust_count+1)
            dyna = Environment(data, None, None, None, *env_params)
            # acts, logps, rew = drl_model(dyna)
            acts1, logps1, rew1, bls = policy_model(dyna, greedy=True) 

            rewards += rew1.mean()
    
    return rewards/iterations, None

def eval_apriori_routes(args, Environment, batch, assignments):
    vehs, custs = batch
    batch_size, nodes_count, d  = custs.size()
    _, n_agent, _ = vehs.size()

    if issubclass(Environment, VRPTW_Environment):
        env_params = [args.pending_cost, args.late_discount]
        if issubclass(Environment, SVRPTW_Environment):
            env_params.extend([args.speed_var, args.late_prob, args.slow_down, args.late_var])

    for n in range(n_agent):
        task_list_end = np.argwhere(assignments[n]==-1)
        assignments[n] = assignments[n][:task_list_end[0][0]] if len(task_list_end) else assignments[n]
    
    task_list = torch.tensor([0] + list(np.concatenate(assignments)))
    mask = custs.new_ones((batch_size, nodes_count), dtype = torch.bool)\
                .scatter(1,torch.tensor([0]+list(task_list))[None,:],0)
    data = PaddedData(vehs=vehs, nodes=custs, cust_mask= mask)
    dyna = Environment(data, None, None, None, *env_params)
    
    rewards = 0
    iterations = 100
    for n_iter in range(iterations):
        dyna.reset()
        routes_it = [_pad_with_zeros(inst_routes) for inst_routes in assignments]
        rew = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor([[next(routes_it[dyna.cur_veh_idx.item()])]], dtype = torch.int64)
            rew.append( dyna.step(cust_idx) )
            # print("cust_idx={}".format(cust_idx))
            # print("rew={}".format(rew))
        rewards += torch.stack(rew).sum(dim = 0).squeeze(-1)
    
    return rewards/iterations, None
