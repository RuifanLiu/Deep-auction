# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:26:18 2020

@author: s313488
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:01:04 2020

@author: s313488
"""
import multiprocessing
import os
import glob
from tkinter import W
from tqdm import tqdm
import numpy as np
from datetime import datetime
from stable_baselines import DQN,A2C
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from multiprocessing import freeze_support, Pool
from customEnv.costum_vec_env import VecNormalize
from customEnv.SubEnvironment import TAEnv_submodular
from customEnv.costum_vec_env import make_ta_env
from customPolicy.common_policies import GATPolicy_sub, A2C_ValueNetwork
from utils.data_util import save_dataset
from utils.func import path_construction

MAX_TASK_NUM = 10

NUM_SAMPLE = 1000

class FilePath:
    def __init__(self, file_path, fig_name, new_fig=True):
        """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param new_tb_log: (bool) whether or not to create a new logging folder for tensorbaord
        """
        self.file_path = file_path
        self.fig_name = fig_name
        self.writer = None
        self.new_fig = new_fig

    def __call__(self):
        if self.file_path is not None:
            latest_run_id = self._get_latest_run_id()
            if self.new_fig:
                latest_run_id = latest_run_id + 1
            save_path = os.path.join(self.file_path, "{}_{}".format(self.fig_name, latest_run_id))
        return save_path

    def _get_latest_run_id(self):
        """
        returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.

        :return: (int) latest run number
        """
        max_run_id = 0
        for path in glob.glob("{}/{}_[0-9]*".format(self.file_path, self.fig_name)):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1].split(".")[0]
            if self.fig_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
                max_run_id = int(ext)
        return max_run_id


def instance(trained_model, policy, RL_algorithm):
    mission_size = np.random.randint(MAX_TASK_NUM)+1
    env0 = TAEnv_submodular(mission_size = mission_size, mission_stochastic_flag='stochastic', 
                        task_duration_stochastic = 'stochastic', time_discount = 0.9,
                        seed=None, maneuver_target=False, dynamic_task = False)

    env1 = make_ta_env(env0, num_env=1, seed=0, use_subprocess=True)
    env1 = VecNormalize(env1, norm_obs=False, norm_reward=False)

    initial_model = eval(RL_algorithm).load(trained_model)
    transfer_model = eval(RL_algorithm)(policy=policy, env=env1)
    par = initial_model.get_parameters()
    transfer_model.load_parameters(par)
    
    obs0 = env1.reset()
    n_iter = 100
    simulation_rewards = []
    for _ in range(n_iter):
        simulation_reward = 0.0
        done,state = False, None
        obs0 = obs0.reshape(env1.observation_space.shape)
        obs = env1.reset(reset_obs=obs0)
        # value = transfer_model.get_value(obs, state=state)
        while not done:
            action, state = transfer_model.predict(obs, state=state,deterministic=True)
            obs, reward, done, info = env1.step(action)
            simulation_reward += reward
            # print(action)
            # print(done)
        simulation_rewards.append(simulation_reward)
    # print(f'Obs:{obs}')
    # print(f'Rewards:{np.mean(simulation_rewards)}')
    # print(f'Value:{value}')
    return {'Input_Obs': obs0, 'input_Value': np.mean(simulation_rewards)}
    zip(obs.tolist(), simulation_reward.tolist())
    #return simulation_reward

def instance_greedy():
    mission_size = np.random.randint(MAX_TASK_NUM)+1
    env0 = TAEnv_submodular(mission_size = mission_size, mission_stochastic_flag='stochastic', 
                        task_duration_stochastic = 'stochastic', time_discount = 0.9,
                        seed=None, maneuver_target=False, dynamic_task = False)

    env1 = make_ta_env(env0, num_env=1, seed=0, use_subprocess=True)
    env1 = VecNormalize(env1, norm_obs=False, norm_reward=False)

    obs0 = env1.reset()

    method = 'greedy'

    path, value = path_construction(obs0, method)

    # n_iter = 100
    # simulationRewards = []
    # for _ in range(n_iter):
    #     simulationReward = 0.0
    #     done, state = False, None
    #     obs = env1.reset(reset_obs=obs0.reshape(env1.observation_space.shape))

    #     for action in path:
    #         obs, reward, done, info = env1.step(np.array([int(action)]))
    #         simulationReward += reward
    #     simulationRewards.append(simulationReward)
    
    # rewards = np.mean(simulationRewards)

    if obs0.shape[-1] == 1:
        rewardsWoFinalTask = 0.0
        value_1 = 0.0

    else:
        obs0_1 = np.copy(obs0)
        obs0_1[:,:,0] = [0, 0, 0, 1, 0]

        path_1, value_1 = path_construction(obs0_1, method)

        # n_iter = 100
        # simulationRewards = []
        # for _ in range(n_iter):
        #     simulationReward = 0.0
        #     done, state = False, None

        #     obs = env1.reset(reset_obs=obs0_1.reshape(env1.observation_space.shape))

        #     for action in path_1:
        #         obs, reward, done, info = env1.step(np.array([int(action)]))
        #         simulationReward += reward
        #     simulationRewards.append(simulationReward)

        # rewardsWoFinalTask = np.mean(simulationRewards)

    # marginReward = rewards - rewardsWoFinalTask
    marginReward = value - value_1
    # assert marginReward>0, 'Break'

    # print(f'Rewards:{np.mean(simulationRewards)}')
    # print(f'Value:{value}')
    return {'Input_Obs': obs0, 'input_Value': marginReward}
    

if __name__ == '__main__':

    model_filename = 'A2C-2022-06-16-11-18-58'
    trained_model = 'Model-A2C-2022-06-16-11-18-58SS-GATPolicy_sub-DOP10'
    policy = 'GATPolicy_sub'
    #last_trained_model = model_string+".zip"
    RL_algorithm = 'A2C'
    
    SavePath = './Simu_result/'+ model_filename
    now = datetime.now() #analyse the time consumption
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S") #time string
    
    num_cpus = os.cpu_count()
    multiprocessing = True

    if not multiprocessing:
        # single thread
        # samples = []
        # for n_sample in range(NUM_SAMPLE):
        #     print('Sample Number:', n_sample)
        #     sample = instance(initial_model, policy)
        #     print(sample)
        #     samples.append(sample)

        with tqdm(range(NUM_SAMPLE), desc = "Generating Training Dataset ", total=NUM_SAMPLE) as pbar:
            samples =[instance(                
                        trained_model,
                        policy,
                        RL_algorithm
                    )
                  for i, _ in enumerate(pbar)]
    else: 
        # parallel processing
        with Pool(num_cpus) as p:
            with tqdm(desc = "Generating Training Dataset", total=NUM_SAMPLE) as pbar:
                samples = [p.apply_async(instance, 
                            args = (trained_model, policy, RL_algorithm), 
                            callback = lambda _:pbar.update()
                            ) for i in range(NUM_SAMPLE)]
                dataset = [res.get() for res in samples]


    # if not multiprocessing:
    #     # single thread
    #     # samples = []
    #     # for n_sample in range(num_samples):
    #     #     print('Sample Number:', n_sample)
    #     #     sample = instance(initial_model, policy)
    #     #     print(sample)
    #     #     samples.append(sample)

    #     with tqdm(range(num_samples), desc = "Generating Training Dataset ", total=num_samples) as pbar:
    #         samples =[instance_greedy()
    #               for i, _ in enumerate(pbar)]
    # else: 
    #     # parallel processing
    #     with Pool(num_cpus) as p:
    #         with tqdm(desc = "Generating Training Dataset", total=num_samples) as pbar:
    #             samples = [p.apply_async(instance_greedy, 
    #                         callback = lambda _:pbar.update()
    #                         ) for i in range(num_samples)]
    #             dataset = [res.get() for res in samples]
    
    filename = 'Train_DataSet-task' +str(MAX_TASK_NUM) + '-size' + str(NUM_SAMPLE) + '_' + dt_string + '.pkl'
    filepath = os.path.join(SavePath, filename)
    save_dataset(dataset, filepath)
