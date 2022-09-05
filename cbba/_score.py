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


PAD_SIZE = 100

def det_reward_func(Environment, custs, veh, task_list):
    batch_size, nodes_count, d  = custs.size()
    mask = custs.new_ones((batch_size, nodes_count), dtype = torch.bool)\
        .scatter(1,torch.tensor([0]+list(task_list))[None,:],0)
    data = PaddedData(vehs=veh[:,None,:], nodes=custs, cust_mask=mask)
    dyna = Environment(data)
    dyna.reset()
    task_list = _pad_with_zeros(task_list)
    rewards = 0
    while not dyna.done:
        cust_idx = dyna.nodes.new_tensor([[next(task_list)]], dtype=torch.int64)
        rewards += dyna.step(cust_idx)
    return rewards, None
        
def sample_reward_func(Environment, custs, veh, task_list, sample=100):
    batch_size, nodes_count, d  = custs.size()
    mask = custs.new_ones((batch_size, nodes_count), dtype = torch.bool)\
        .scatter(1,torch.tensor([0]+list(task_list))[None,:],0)
    data = PaddedData(vehs=veh[:,None,:], nodes=custs, cust_mask= mask)
    dyna = Environment(data)
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


def Scoring_CalcScore_Original(Environment, n, custs, vehs, taskCurr, taskPrev, timePrev, taskNext, timeNext, robust_greedy=False):
    oldTaskList = taskPrev + taskNext
    newTaskList = taskPrev + [taskCurr] + taskNext

    if not robust_greedy:
        output_reward_new, output_time_new = det_reward_func(Environment, custs, vehs[:,n,:], newTaskList)
    else:
        output_reward_new, output_time_new = sample_reward_func(Environment, custs, vehs[:,n,:], newTaskList)
        
    # print('oldTaskList:',oldTaskList)
    # print('newTaskList:',newTaskList)
    # print('output_reward_new:',output_reward_new)
    # print('\n')
   
    minStart, maxStart = 0, 0
    output_reward_old = []
    return output_reward_old, output_reward_new, minStart, maxStart

def Scoring_CalcScore_DNN(Environment, n, custs, vehs, value_model, unAllocatedTask, taskPrev, timePrev, taskNext, timeNext):   
    
    oldTaskList = [0] + taskPrev + taskNext
    newTaskListList = [[0] + taskPrev + [m] + taskNext for m in unAllocatedTask]

    nodes = torch.cat([custs[:,newTaskList] for newTaskList in newTaskListList])
    vehs = torch.cat([vehs[:,n,:][:,None,:] for _ in newTaskListList])
    data = PaddedData(vehs=vehs, nodes=nodes, padding_size=value_model.cust_count+1)
    output_reward_new = value_model.eval_init(Environment(data))
   
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



