# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:55:50 2020

@author: s313488
"""
from matplotlib.style import available
import torch
import numpy as np
import time
from cbba._data_util import data_padding, data_padding_zeros
from cbba._data_util import PaddedData, _padding_with_zeros
from cbba._mdp_utils import MDP
from routing_model.utils._misc import _pad_with_zeros
from problems import *


PAD_SIZE = 100

def det_reward_func(args, custs, veh, task_list):

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
        late = np.clip( arv - dest[5], 0, None)
        vehicles[:2] = dest[:2]
        vehicles[2] -= dest[2]
        vehicles[4] = arv + dest[6]

        # rewards += -dist - np.clip(args.late_cost*late, 0, args.pending_cost)
        finish = (arv <= dest[5])
        rewards += dest[3]* finish - args.pending_cost* (1-finish) #* np.exp(-args.late_discount*late)
        if cust_idx == 0:
            done = True
    return rewards, None

def sample_reward_func(args, custs, veh, task_list, sample=1000):
    task_list = task_list + [0]
    batch_size, nodes_count, d  = custs.size()
    custs = np.copy(custs[0,:,:])

    total_reward = 0
    for n_sample in range(sample):
        vehicles = np.zeros((5,))
        vehicles[:2] = custs[0,:2]
        vehicles[2:4] = np.copy(veh)
        rewards, done = 0, False
        for cust_idx in task_list:
            dest = custs[cust_idx,:]
            dist = np.linalg.norm(dest[:2]-vehicles[:2], axis=-1)
            veh_speed = np.random.normal(loc=vehicles[3], scale=args.speed_var)
            tt = dist / veh_speed
            arv = np.max([vehicles[4] + tt, dest[4]])
            late = np.clip( arv - dest[5], 0, None)
            vehicles[:2] = dest[:2]
            vehicles[2] -= dest[2]
            vehicles[4] = arv + dest[6]

            # rewards += -dist - np.clip(args.late_cost*late, 0, args.pending_cost)
            finish = (arv <= dest[5])
            rewards += dest[3]* finish - args.pending_cost* (1-finish) #* np.exp(-args.late_discount*late)
            if cust_idx == 0:
                done = True
        total_reward += rewards
    
    return total_reward/sample, None
        


def Scoring_CalcScore_Original(args, Environment, n, custs, vehs, m, taskPrev, timePrev, taskNext, timeNext, robust_greedy=False):  
    newTaskList = taskPrev + [m] +taskNext

    if not robust_greedy:
        output_reward_new, output_time_new = det_reward_func(args, custs, vehs[:,n,:], newTaskList)
    else:
        output_reward_new, output_time_new = sample_reward_func(args, custs, vehs[:,n,:], newTaskList)
    # t_end = time.perf_counter()
    # print('cal_time:',t_end-t_start)
    # print('new task:',m)
    # print('newTaskList:',newTaskList)
    # print('output_reward_new:',output_reward_new)
    # print('\n')

    minStart, maxStart = 0, 0
    output_reward_old = []
    return output_reward_old, output_reward_new, minStart, maxStart

def Scoring_CalcScore_DNN(args, Environment, n, custs, vehs, value_model, unAllocatedTask, taskPrev, timePrev, taskNext, timeNext):   
    
    oldTaskList = [0] + taskPrev + taskNext
    newTaskListList = [[0] + taskPrev + [m] + taskNext for m in unAllocatedTask]

    if issubclass(Environment, VRPTW_Environment):
        env_params = [args.pending_cost, args.late_cost]
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

def Scoring_CalcScore_MDP(args, MDP, n, custs, vehs, Value, unAllocatedTask, taskPrev, timePrev, taskNext, timeNext):   
    
    oldTaskList = taskPrev + taskNext
    newTaskListList = [taskPrev + [m] + taskNext for m in unAllocatedTask]

    batch_size, nodes_count, node_state_size  = custs.size()
    output_reward_new = []
    for newTaskList in newTaskListList:
        available_task = np.zeros(nodes_count)
        available_task[newTaskList] = 1
        state = {
                'time': 0,
                'available_task': available_task,
                'cur_node': 0
            }
        idx = MDP.state_to_idx(state)
        value = Value[idx]
        output_reward_new.append(value)
    
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



