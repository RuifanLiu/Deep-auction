# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:55:50 2020

@author: s313488
"""

import numpy as np
import time
import tensorflow as tf
# from utils.GraphTransfer import obs2Graph
# from stellargraph.mapper import PaddedGraphGenerator
from customEnv.costum_vec_env import VecNormalize, make_ta_env
from stable_baselines import A2C
from customPolicy.common_policies import GATPolicy_sub
from utils.data_util import data_padding, data_padding_zeros


PAD_SIZE = 100

def reward_function(obs, task_list, robust_greedy):
    task_rewards = np.copy(obs[0,:]) # 0 reward 1 task_duration 2 start_time 3 visit_flag
    task_durations = np.copy(obs[1,:])
    task_startTime = np.copy(obs[2,:])
    task_visitflag = np.copy(obs[3,:]) # 0 pending 1 visited
    system_time = np.copy(obs[4,:])[0]

    time_discount = 0.9
    UNCERTAINTY_sigma = 1.0

    #robust_greedy = 1
    if robust_greedy:
        n_sample = 1000
        rewards = 0
        system_time = [system_time]
        for ii in range(len(task_list)):
            task_no = int(task_list[ii])
            task_duration_mean = task_durations[task_no]
            task_duration = np.random.normal(task_duration_mean, UNCERTAINTY_sigma, n_sample)
            task_duration = np.clip(task_duration, 0, task_duration_mean*2)

            task_executionTime = np.array([np.max([task_startTime[task_no], sys_time]) for sys_time in system_time])
            task_reward = task_rewards[task_no]*np.exp(-time_discount*(task_executionTime-task_startTime[task_no]))
            reward = task_reward if not task_visitflag[task_no] else 0.0    

            system_time = task_executionTime + task_duration
            task_visitflag[task_no] = 1 # task is masked once visited

            rewards = rewards + reward
            
        output_reward = np.mean(rewards)
        output_time = np.mean(system_time)
    else:
        rewards = 0

        for ii in range(len(task_list)):
            task_no = int(task_list[ii])
            
            task_duration_mean = task_durations[task_no]
            task_duration = task_duration_mean

            task_executionTime = np.max([task_startTime[task_no], system_time])
            task_reward = task_rewards[task_no]*np.exp(-time_discount*(task_executionTime-task_startTime[task_no]))
            reward = task_reward if not task_visitflag[task_no] else 0.0    

            system_time = task_executionTime + task_duration
            task_visitflag[task_no] = 1 # task is masked once visited

            rewards = rewards + reward
            
        output_reward = rewards
        output_time = system_time

    return output_reward, output_time

def Scoring_CalcScore_Original(obs, taskCurr, taskPrev, timePrev, taskNext, timeNext, robust_greedy=False):
    oldTaskList = taskPrev + taskNext
    newTaskList = taskPrev + [taskCurr] + taskNext

    # output_reward_old, output_time_old = reward_function(obs, oldTaskList, robust_greedy)
    output_reward_new, output_time_new = reward_function(obs, newTaskList, robust_greedy)
        
    # print('oldTaskList:',oldTaskList)
    # print('newTaskList:',newTaskList)
    # print('output_reward_old:',output_reward_old)
    # print('output_reward_new:',output_reward_new)
    # print('\n')
   
    minStart, maxStart = 0, 0
    output_reward_old = []
    return output_reward_old, output_reward_new, minStart, maxStart

class reward_function_DRL():
    def __init__(self, env0, initial_model):

        env1 = make_ta_env(env0, num_env=1, seed=0, use_subprocess=True)
        env1 = VecNormalize(env1,norm_obs=False, norm_reward=False)
        transfer_model = A2C('GATPolicy_sub', env=env1)
        par = initial_model.get_parameters()
        transfer_model.load_parameters(par)

        self.env0 = env0
        self.env1 = env1
        self.transfer_model = transfer_model

    def __call__(self, obs):

        iterations = 100
        actual_rewards = []
        for n_iter in range(iterations):
            
            obs = obs.reshape(self.env0.observation_space.shape)
            obs = self.env1.reset(obs)

            simulation_reward = 0.0
            simulation_length = 0
            simulation_delayTime = 0.0
            done,state = False, None

            while not done:
                t_start = time.perf_counter()
                action, state = self.transfer_model.predict(obs, state=state, deterministic=True)
                t_end = time.perf_counter()
                cal_time = t_end-t_start
                
                obs, reward, done, info = self.env1.step(action)
                simulation_reward += reward
                simulation_delayTime += info[0]
                simulation_length += 1
            
            # record the reward for each agent and each iteration 
            # axis0: agent; axis1: simulation_iteration
            actual_rewards.append(simulation_reward) 


        return np.mean(actual_rewards)

def Scoring_CalcScore_DRL(obs, loaded_model, unAllocatedTask, taskPrev, timePrev, taskNext, timeNext, reward_function_DRL):
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

    # data padding
    pad_size = node_size
    oldObs = data_padding_zeros(oldObs, pad_size)
    newObsList = [data_padding_zeros(newObs, pad_size) for newObs in newObsList]   

    if len(oldTaskList):
        output_reward_old = reward_function_DRL(oldObs)
        output_reward_new = [reward_function_DRL(newObs) for newObs in newObsList]
        marginalReward = [i-output_reward_old for i in output_reward_new]
    else:
        output_reward_old = 0
        output_reward_new = [reward_function_DRL(newObs) for newObs in newObsList]
        marginalReward = [i-output_reward_old for i in output_reward_new]

    # print('oldTaskList:',oldTaskList)
    # print('unAllocatedTask:',unAllocatedTask)
    # print('output_reward_old:',output_reward_old)
    # print('output_reward_new:',output_reward_new)
    # print('max marginalReward:',np.max(marginalReward))
    # print('\n')
    minStart, maxStart = 0, 0
    return marginalReward, minStart, maxStart    
    
def Scoring_CalcScore_DNN(obs, value_mode, unAllocatedTask, taskPrev, timePrev, taskNext, timeNext):   
    
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
        output_reward_new = [value_mode.predict(np.expand_dims(newObs,0)) for newObs in newObsList]
        # marginalReward = [i-output_reward_old for i in output_reward_new]
    else:
        # output_reward_old = 0
        output_reward_new = [value_mode.predict(np.expand_dims(newObs,0)) for newObs in newObsList]
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

def Scoring_CalcScore_Maxmin_DNN(obs, value_mode, value_model2, unAllocatedTask, taskPrev, timePrev, taskNext, timeNext):   
    
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


def actualReward(env0, multiObs, routingMethod, initial_model, CBBA_Assignments, CBBA_Class, loaded_model):
    n_agent = len(CBBA_Assignments)
    if routingMethod == 'Original_Sequence_Order':
        env1 = make_ta_env(env0, num_env=1, seed=0, use_subprocess=True)
        env1 = VecNormalize(env1,norm_obs=False, norm_reward=False)
        
        iterations = 100
        Actual_rewards = []
        
        DelayTimes = []
        for n_iter in range(iterations):
            agent_rewards, agent_lengths = [], []
            agent_delayTimes = []
            Estimate_rewards = []
            

            for n in range(n_agent):
                task_list_end = np.argwhere(CBBA_Assignments[n]==-1)
                task_list = CBBA_Assignments[n][:task_list_end[0][0]] if len(task_list_end) else CBBA_Assignments[n]

                obs0 = np.copy(multiObs[n])
                obs0 = obs0.reshape(env0.observation_space.shape)
                obs = env1.reset(obs0)

                simulation_reward = 0.0
                simulation_length = 0
                simulation_delayTime = 0.0

                task_to_execute = task_list
                done, state = False, None  
                while len(task_to_execute) and not done:
                    task_no = task_to_execute[0]
                    task_to_execute = task_to_execute[1:]
                    action = np.array([int(task_no)])
                    
                    #check the budget constraint
                    obs = obs.reshape(env1.observation_space.shape)
                    obs, reward, done, info = env1.step(action) # info - delayed time
                    simulation_reward += reward
                    simulation_delayTime += info[0]
                    simulation_length += 1

                    # print(f'Action: {action} \t Reward: {reward} \t Done: {done} \t Simu_length: {simulation_length} \t Time: {obs[0][4,0]}')

                agent_rewards.append(simulation_reward)
                agent_lengths.append(simulation_length)
                agent_delayTimes.append(simulation_delayTime)
                # expected_reward = sum(expected_reward)
                # episode_success_percentage = simulation_success/n_simulations

            Actual_rewards.append(sum(agent_rewards))
            DelayTimes.append(sum(agent_delayTimes))

        return np.mean(Actual_rewards), np.mean(DelayTimes)

    elif routingMethod == 'DRL_planner': 
        env1 = make_ta_env(env0, num_env=1, seed=0, use_subprocess=True)
        env1 = VecNormalize(env1,norm_obs=False, norm_reward=False)
        transfer_model = A2C('GATPolicy_sub', env=env1)
        par = initial_model.get_parameters()
        transfer_model.load_parameters(par)

        iterations = 100
        Actual_rewards = []
        actual_rewards = []
        DelayTimes = []
        for n_iter in range(iterations):
            agent_rewards, agent_lengths = [], [] 
            agent_delayTimes = []
            Estimate_rewards = []
            
            for n in range(n_agent):
                obs0 = np.copy(multiObs[n])
                obs0 = obs0.reshape(env0.observation_space.shape)
            
                task_list_end = np.argwhere(CBBA_Assignments[n]==-1)
                task_list = CBBA_Assignments[n][:task_list_end[0][0]] if len(task_list_end) else CBBA_Assignments[n]
                
                # task_remove = np.setdiff1d(np.array(range(CBBA_Class.CBBA_Params_M)), task_list)
                obs_task = obs0[:,task_list] 
                obs0 = data_padding(obs_task, pad_size=env1.observation_space.shape[1])

                obs_estimate = data_padding_zeros(obs_task, pad_size=PAD_SIZE)
                estimate_reward = loaded_model.predict(np.expand_dims(obs_estimate,0))
                Estimate_rewards.append(estimate_reward)

                obs = env1.reset(obs0)

                simulation_reward = 0.0
                simulation_length = 0
                simulation_delayTime = 0.0
                done,state = False, None

                _, expected_reward, _, _ = transfer_model.step(obs, state, done, deterministic=True)

                while not done:
                    t_start = time.perf_counter()
                    action, state = transfer_model.predict(obs, state=state, deterministic=True)
                    t_end = time.perf_counter()
                    cal_time = t_end-t_start
                    
                    obs, reward, done, info = env1.step(action)
                    simulation_reward += reward
                    simulation_delayTime += info[0]
                    simulation_length += 1

                agent_rewards.append(simulation_reward)
                agent_lengths.append(simulation_length)
                agent_delayTimes.append(simulation_delayTime)
            
            # record the reward for each agent and each iteration 
            # axis0: agent; axis1: simulation_iteration
            actual_rewards.append(np.concatenate(agent_rewards).reshape(n_agent,1)) 

        Actual_rewards=np.concatenate(actual_rewards,axis=1)
        DelayTimes.append(sum(agent_delayTimes))

        print('Estimate reward for each agent:')
        print(np.concatenate(Estimate_rewards))
        print('Actual reward for each agent:')
        print(np.mean(Actual_rewards,axis=1))

        return np.mean(np.sum(Actual_rewards,axis=0)), np.mean(DelayTimes)

