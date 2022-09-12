# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:55:50 2020

@author: s313488
"""
import torch
import numpy as np
import time
from cbba._data_util import data_padding, data_padding_zeros
from cbba._data_util import PaddedData, _padding_with_zeros
from routing_model.utils._misc import _pad_with_zeros
from problems import *


PAD_SIZE = 100

def det_reward_func(args, custs, veh, task_list):

    # if issubclass(Environment, VRPTW_Environment):
    #     env_params = [args.pending_cost, args.late_discount]
    #     if issubclass(Environment, SVRPTW_Environment):
    #         env_params.extend([args.speed_var, args.late_prob, args.slow_down, args.late_var])

    # batch_size, nodes_count, d  = custs.size()
    # mask = custs.new_ones((batch_size, nodes_count), dtype = torch.bool)\
    #     .scatter(1,torch.tensor([0]+list(task_list))[None,:],0)
    # data = PaddedData(vehs=veh[:,None,:], nodes=custs, cust_mask=mask)

    # dyna = Environment(data, None, None, None, *env_params)
    # dyna.reset()
    # task_list = _pad_with_zeros(task_list)
    # rewards = 0
    # while not dyna.done:
    #     cust_idx = dyna.nodes.new_tensor([[next(task_list)]], dtype=torch.int64)
    #     rewards += dyna.step(cust_idx)
    # # return rewards, None


    # task_list = _pad_with_zeros(task_list)
    # rewards = 0
    # done = False
    # batch_size, nodes_count, d  = custs.size()
    # vehicles = custs.new_zeros((batch_size, 1, 5))
    # vehicles[:,:,:2] = custs[:,0:1,:2]
    # vehicles[:,:,2:4] = veh

    # while not done:
    #     cust_idx = next(task_list)
    #     dest = custs[:,cust_idx,:][:,None,:]
    #     dist = torch.pairwise_distance(vehicles[:,0,:2], dest[:,0,:2], keepdim = True)
    #     tt = dist / vehicles[:,0,3]
    #     arv = torch.max(vehicles[:,:,4] + tt, dest[:,:,4])
    #     late = ( arv - dest[:,:,4] ).clamp_(min = 0)
    #     vehicles[:,:,:2] = dest[:,:,:2]
    #     vehicles[:,:,2] -= dest[:,:,2]
    #     vehicles[:,:,4] = arv + dest[:,:,6]

    #     finish = torch.le(vehicles[:,:,4],dest[:,:,5])
    #     rewards += finish * dest[:,:,3] * torch.exp(-args.late_discount*late)
        
    #     if cust_idx == 0:
    #         done = True

    t_start = time.perf_counter()
    task_list = task_list + [0]
    rewards, done = 0, False
    batch_size, nodes_count, d  = custs.size()
    custs = np.copy(custs[0,:,:])
    vehicles = np.zeros((5,))
    vehicles[:2] = custs[0,:2]
    vehicles[2:4] = np.copy(veh)

    for cust_idx in task_list:
        dest = custs[cust_idx,:]
        dist = np.linalg.norm(dest[:2]-vehicles[:2], axis=-1)
        tt = dist / vehicles[3]
        arv = np.max([vehicles[4] + tt, dest[4]])
        late = np.clip( arv - dest[4], 0, None)
        vehicles[:2] = dest[:2]
        vehicles[2] -= dest[2]
        vehicles[4] = arv + dest[6]
        finish = (vehicles[4] <= dest[5])
        rewards += finish * dest[3] * np.exp(-args.late_discount*late)
        if cust_idx == 0:
            done = True
    return rewards, None

def sample_reward_func(args, Environment, custs, veh, task_list, sample=100):

    if issubclass(Environment, VRPTW_Environment):
        env_params = [args.pending_cost, args.late_discount]
        if issubclass(Environment, SVRPTW_Environment):
            env_params.extend([args.speed_var, args.late_prob, args.slow_down, args.late_var])

    batch_size, nodes_count, d  = custs.size()
    mask = custs.new_ones((batch_size, nodes_count), dtype = torch.bool)\
        .scatter(1,torch.tensor([0]+list(task_list))[None,:],0)
    data = PaddedData(vehs=veh[:,None,:], nodes=custs, cust_mask= mask)
    dyna = Environment(data, None, None, None, *env_params)
    task_list = _pad_with_zeros(task_list)  
    rewards = 0 
    for n_sample in range(sample):
        dyna.reset()
        rew = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor([[next(task_list)]], dtype=torch.int64)
            rew.append( dyna.step(cust_idx) )
        rewards += torch.stack(rew).sum(dim = 0).squeeze(-1)
    return rewards/sample, None


def Scoring_CalcScore_Original(args, Environment, n, custs, vehs, m, taskPrev, timePrev, taskNext, timeNext, robust_greedy=False):  
    newTaskList = taskPrev + [m] +taskNext

    if not robust_greedy:
        output_reward_new, output_time_new = det_reward_func(args, custs, vehs[:,n,:], newTaskList)
    else:
        output_reward_new, output_time_new = sample_reward_func(args, Environment, custs, vehs[:,n,:], newTaskList)
    # t_end = time.perf_counter()
    # print('oldTaskList:',oldTaskList)
    # print('newTaskList:',newTaskList)
    # print('cal_time:',t_end-t_start)
    # print('output_reward_new:',output_reward_new)
    # print('\n')

    minStart, maxStart = 0, 0
    output_reward_old = []
    return output_reward_old, output_reward_new, minStart, maxStart

def Scoring_CalcScore_DNN(args, Environment, n, custs, vehs, value_model, unAllocatedTask, taskPrev, timePrev, taskNext, timeNext):   
    
    oldTaskList = [0] + taskPrev + taskNext
    newTaskListList = [[0] + taskPrev + [m] + taskNext for m in unAllocatedTask]

    if issubclass(Environment, VRPTW_Environment):
        env_params = [args.pending_cost, args.late_discount]
        if issubclass(Environment, SVRPTW_Environment):
            env_params.extend([args.speed_var, args.late_prob, args.slow_down, args.late_var])

    nodes = torch.cat([custs[:,newTaskList] for newTaskList in newTaskListList])
    vehs = torch.cat([vehs[:,n,:][:,None,:] for _ in newTaskListList])
    data = PaddedData(vehs=vehs, nodes=nodes, padding_size=value_model.cust_count+1)
    output_reward_new = value_model.eval_init(Environment(data, None, None, None, *env_params))
   
    # print('oldTaskList:',oldTaskList)
    # print('unAllocatedTask:',unAllocatedTask)
    # print('output_reward_new:',output_reward_new)
    # print('\n')
    minStart, maxStart = 0, 0
    output_reward_old = []
    return output_reward_old, output_reward_new, minStart, maxStart

def Scoring_CalcScore_Maxmin_DNN(custs, veh, value_mode, value_model2, unAllocatedTask, taskPrev, timePrev, taskNext, timeNext):   
    
    oldTaskList = taskPrev + taskNext
    oldTask_target = oldTaskList
    newTaskListList = []
    for m in unAllocatedTask:
        newTaskList = taskPrev + [m] + taskNext
        newTaskList = newTaskList
        newTaskListList.append(newTaskList)
    
    feat_size, node_size = obs.shape
    nodeFeatures = obs
    oldObs = nodeFeatures[:,oldTask_target]
    
    newObsList = []
    for newTaskList in newTaskListList:
        newObs = nodeFeatures[:,newTaskList]
        newObsList.append(newObs)
    
    oldObs = data_padding_zeros(oldObs, PAD_SIZE)
    newObsList = [data_padding_zeros(newObs, PAD_SIZE) for newObs in newObsList]   
    if len(oldTaskList):
        # output_reward_old = value_mode.predict(np.expand_dims(oldObs,0))
        reward1 = [value_mode.predict(np.expand_dims(newObs,0)) for newObs in newObsList]
        reward2 = [value_model2.predict(np.expand_dims(newObs,0)) for newObs in newObsList]
        output_reward_new = [min([reward1[i], reward2[i]]) for i in range(len(newObsList))]

        # marginalReward = [i-output_reward_old for i in output_reward_new]
    else:
        # output_reward_old = 0
        reward1 = [value_mode.predict(np.expand_dims(newObs,0)) for newObs in newObsList]
        reward2 = [value_model2.predict(np.expand_dims(newObs,0)) for newObs in newObsList]
        output_reward_new = [min([reward1[i], reward2[i]]) for i in range(len(newObsList))]
        # marginalReward = [i-output_reward_old for i in output_reward_new]

    # print('oldTaskList:',oldTaskList)
    # print('unAllocatedTask:',unAllocatedTask)
    # print('output_reward_old:',output_reward_old)
    # print('output_reward_new:',output_reward_new)
    # print('max marginalReward:',np.max(marginalReward))
    # print('\n')
    minStart, maxStart = 0, 0
    output_reward_old = []
    return output_reward_old, output_reward_new, minStart, maxStart



