# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:56:48 2020

@author: s313488
"""
import os
from re import I
import time
from datetime import datetime
import logging
import numpy as np
import tensorflow as tf
from stable_baselines import A2C

from customEnv import SubEnvironment
from utils.networks import MultiHeadAttention
from customPolicy.common_policies import A2C_ValueNetwork
from customEnv.costum_vec_env import VecNormalize, make_ta_env


from CBBA import CBBA
from CBBA_rewardFun import actualReward


if __name__=='__main__':

    ### value model - using supervised learning
    value_estimate_model = 'Saved_model/my_model-20-07-2022-22-00-52'
    value_model = tf.keras.models.load_model( value_estimate_model,
                                            custom_objects={'MultiHeadAttention': MultiHeadAttention})
    value_estimate_model = 'Saved_model/my_model-21-08-2022-17-23-39'
    value_model2 = tf.keras.models.load_model( value_estimate_model,
                                            custom_objects={'MultiHeadAttention': MultiHeadAttention})
    # ### value model - using value network in A2C networks
    # value_estimate_model = 'ValueNN_model/ValueNetwork-2022-07-08-11-13-40.zip'
    # initial_model = A2C_ValueNetwork.load(value_estimate_model)

    # value_env0 = SubEnvironment.TAEnv_submodular(mission_size = 20,
    #                                     mission_stochastic_flag='stochastic', 
    #                                     task_duration_stochastic = 'stochastic')
    # value_env1 = make_ta_env(value_env0, num_env=1, seed=0, use_subprocess=True)
    # value_env1 = VecNormalize(value_env1,norm_obs=False, norm_reward=False)

    # loaded_model = A2C_ValueNetwork('GATPolicy_sub', env=value_env1, batch_size = 1.0)
    # par = initial_model.get_parameters()
    # loaded_model.load_parameters(par)

    # load the dynamic path planning model
    dt_string = 'A2C-2022-06-16-11-18-58'
    last_trained_model='Model-A2C-2022-06-16-11-18-58SS-GATPolicy_sub-DOP10'
    RL_algorithm = 'A2C'
    DRL_model = eval(RL_algorithm).load(last_trained_model)

    file_path = './Log_files'
    now = datetime.now() #analyse the time consumption
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S") #time string
    file_name = 'log-'+dt_string+'.txt'

    file_name = os.path.join(file_path, file_name)
    logging.basicConfig(filename=file_name,
                level=logging.DEBUG,
                format='%(message)s',  #%(levelname)s:
                datefmt='%I:%M:%S')
    
    logging.info('Value Estimate Model:'+value_estimate_model)
    logging.info('DRL Route Planner Model:'+last_trained_model)

    ActualReweards1, ActualReweards2 = [], []
    ExpectReweards1, ExpectReweards2 = [], []
    ActualReweards3, ExpectReweards3 = [], []
    ActualReweards4, ActualReweards5 = [], []

    CalTimes1, CalTimes2, CalTimes3=[],[],[]
    n_agent = 5
    mission_size = 20
    for n_agent in range(2,11,1):
    # for mission_size in range(100,101,10):
        #loaded_model = []
        # ---------------------------------------------------------------------
        #  Define sample scenario
        # ---------------------------------------------------------------------
        #mission_size = 5*n_agent
        env0 = SubEnvironment.TAEnv_submodular(mission_size = mission_size,
                                        mission_stochastic_flag='stochastic', 
                                        task_duration_stochastic = 'stochastic',
                                        time_discount = 0.9,
                                        seed=None, maneuver_target=False, dynamic_task = False) 
        attrs = env0.__dict__
        for key in attrs:
            logging.info(key + ': ' + str(attrs[key]))
            
        # n_agent = 2
        iterations = 5

        logging.info('Mission size:'+ str(mission_size))
        logging.info('Agent number:'+ str(n_agent))
        logging.info('Iterations:'+ str(iterations))
        logging.info('\n')

        actualReweards1, actualReweards2, actualReweards3 = [], [], []
        expectReweards1, expectReweards2, expectReweards3 = [], [], []
        actualReweards4 = []
        actualReweards5, expectReweards5 = [], []

        delayTimes1, delayTimes2, delayTimes3, delayTimes4, delayTimes5 = [],[],[],[],[]
        CalTime1, CalTime2, CalTime3=[],[],[]
        CalTime_ddnn = []
        actualReweards_ddnn = []
        expectReweards_ddnn = []
        delayTimes_ddnn = []
        for n_iter in range(iterations):

            logging.info(f'[Iteration {n_iter}]')

            multiObs = []
            for n in range(n_agent):
                obs = env0.reset()
                multiObs.append(obs)
            
            Graph = 1-np.eye(n_agent)
            
            # --------------------------------------------------------------------
            #  Task Assignment & Evaluation
            # --------------------------------------------------------------------
            # scorefun = 'Scoring_CalcScore_DNN'
            # t_start = time.perf_counter()
            # CBBA_Class1 = CBBA(multiObs, scorefun, Graph, value_model=value_model, DRL_model=DRL_model, env=env0, value_model2 = value_model2)
            # CBBA_Assignments1, Total_Score1 = CBBA_Class1.CBBA_Main()
            # t_end = time.perf_counter()
            # cal_time1 = t_end-t_start
            # CalTime1.append(cal_time1)
            
            # routingMethod = 'DRL_planner'
            # actual_reward1, delay1 = actualReward(env0, multiObs, routingMethod, DRL_model, CBBA_Assignments1, CBBA_Class1, value_model)
            # print('DNN-CBBA Expected Score:', Total_Score1)
            # print('DNN-CBBA Actual Score:', actual_reward1)
            # actualReweards1.append(actual_reward1)
            # expectReweards1.append(Total_Score1)
            # delayTimes1.append(delay1)

            # logging.info('DNN-CBBA Assignment:'+str(CBBA_Assignments1))
            # logging.info('DNN-CBBA Expected Score:'+str(Total_Score1))
            # logging.info('DNN-CBBA Actual Score:'+str(actual_reward1))
            # logging.info('DNN-CBBA Computing Time:'+str(cal_time1))
            # logging.info('DNN-CBBA Task Delayed Time:'+str(delay1))


            # --------------------------------------------------------------------
            #  Task Assignment & Evaluation Double DNN-CBBA
            # --------------------------------------------------------------------
            scorefun = 'Scoring_CalcScore_Maxmin_DNN'
            t_start = time.perf_counter()
            CBBA_Class_ddnn = CBBA(multiObs, scorefun, Graph, value_model=value_model, DRL_model=DRL_model, env=env0, value_model2 = value_model2)
            CBBA_Assignments_ddnn, Total_Score_ddnn = CBBA_Class_ddnn.CBBA_Main()
            t_end = time.perf_counter()
            cal_time_ddnn = t_end-t_start
            CalTime_ddnn.append(cal_time_ddnn)
            
            routingMethod = 'DRL_planner'
            actual_reward_ddnn, delay_ddnn = actualReward(env0, multiObs, routingMethod, DRL_model, CBBA_Assignments_ddnn, CBBA_Class_ddnn, value_model)
            print('DNN-CBBA Expected Score:', Total_Score_ddnn)
            print('DNN-CBBA Actual Score:', actual_reward_ddnn)
            actualReweards_ddnn.append(actual_reward_ddnn)
            expectReweards_ddnn.append(Total_Score_ddnn)
            delayTimes_ddnn.append(delay_ddnn)

            logging.info('Double DNN-CBBA Assignment:'+str(CBBA_Assignments_ddnn))
            logging.info('Double DNN-CBBA Expected Score:'+str(Total_Score_ddnn))
            logging.info('Double DNN-CBBA Actual Score:'+str(actual_reward_ddnn))
            logging.info('Double DNN-CBBA Computing Time:'+str(cal_time_ddnn))
            logging.info('Double DNN-CBBA Task Delayed Time:'+str(delay_ddnn))
            
            # --------------------------------------------------------------------
            #  Task Assignment & Evaluation Original CBBA
            # --------------------------------------------------------------------
            # scorefun = 'Scoring_CalcScore_Original'
            # t_start = time.perf_counter()
            # CBBA_Class2 = CBBA(multiObs, scorefun, Graph)
            # CBBA_Assignments2, Total_Score2 = CBBA_Class2.CBBA_Main()
            # t_end = time.perf_counter()
            # cal_time2 = t_end-t_start
            # CalTime2.append(cal_time2)
            
            # routingMethod = 'Original_Sequence_Order'
            # actual_reward2, delay2 = actualReward(env0, multiObs, routingMethod, DRL_model, CBBA_Assignments2, CBBA_Class2, value_model)
            # print('CBBA Expected Score:', Total_Score2)
            # print('CBBA Actual Score:', actual_reward2)
            # actualReweards2.append(actual_reward2)
            # expectReweards2.append(Total_Score2)
            # delayTimes2.append(delay2)
            
            # logging.info('CBBA Assignment:'+str(CBBA_Assignments2))
            # logging.info('CBBA Expected Score:'+str(Total_Score2))
            # logging.info('CBBA Actual Score:'+str(actual_reward2))
            # logging.info('CBBA Computing Time:'+str(cal_time2))
            # logging.info('CBBA Task Delayed Time:'+str(delay2))

            # routingMethod = 'DRL_planner'
            # actual_reward4, delay4 = actualReward(env0, multiObs, routingMethod, DRL_model, CBBA_Assignments2, CBBA_Class2, value_model)
            # actualReweards4.append(actual_reward4)
            # delayTimes4.append(delay4)

            # logging.info('CBBA Actual Score - DRL_planner:'+str(actual_reward4))
            # logging.info('CBBA Task Delayed Time  - DRL_planner:'+str(delay4))
            
            # --------------------------------------------------------------------
            #  Task Assignment & Evaluation DNN CBBA without DRL planner
            # --------------------------------------------------------------------

            # scorefun = 'Scoring_CalcScore_DNN_with_Insertion_Order'
            # t_start = time.perf_counter()
            # CBBA_Class3 = CBBA(multiObs, scorefun, Graph, loaded_model)
            # CBBA_Assignments3, Total_Score3 = CBBA_Class3.CBBA_Main()
            # t_end = time.perf_counter()
            # cal_time3 = t_end-t_start
            # CalTime3.append(cal_time3)
            
            # routingMethod = 'Original_Sequence_Order'
            # actual_reward5, delay5 = actualReward(env0, multiObs, routingMethod, initial_model, CBBA_Assignments3, CBBA_Class3, loaded_model)
            # print('DNN-CBBA-wo-DRL Expected Score:', Total_Score3)
            # print('DNN-CBBA-wo-DRL Actual Score:', actual_reward5)
            # actualReweards5.append(actual_reward5)
            # expectReweards5.append(Total_Score3)
            # delayTimes5.append(delay5)

            # logging.info('DNN-CBBA-wo-DRL Assignment:'+str(CBBA_Assignments3))
            # logging.info('DNN-CBBA-wo-DRL Expected Score:'+str(Total_Score3))
            # logging.info('DNN-CBBA-wo-DRL Actual Score:'+str(actual_reward5))
            # logging.info('DNN-CBBA-wo-DRL Computing Time:'+str(cal_time3))
            # logging.info('DNN-CBBA-wo-DRL Task Delayed Time:'+str(delay5))
            
            # --------------------------------------------------------------------
            #  Task Assignment & Evaluation Robust CBBA
            # --------------------------------------------------------------------
            # scorefun = 'Scoring_CalcScore_Robust'
            # t_start = time.perf_counter()
            # CBBA_Class3 = CBBA(multiObs, scorefun, Graph, loaded_model)
            # CBBA_Assignments3, Total_Score3 = CBBA_Class3.CBBA_Main()
            # t_end = time.perf_counter()
            # cal_time3 = t_end-t_start
            # CalTime3.append(cal_time3)
            
            # routingMethod = 'Original_Sequence_Order'
            # actual_reward3 = actualReward(env0, multiObs, routingMethod, DRL_model, CBBA_Assignments3, CBBA_Class3, loaded_model)
            # print('Robust CBBA Expected Score:', Total_Score3)
            # print('Robust CBBA Actual Score:', actual_reward3)
            # actualReweards3.append(actual_reward3)
            # expectReweards3.append(Total_Score3)

            # logging.info('Robust CBBA Assignment:'+str(CBBA_Assignments3))
            # logging.info('Robust CBBA Expected Score:'+str(Total_Score3))
            # logging.info('Robust CBBA Actual Score:'+str(actual_reward3))
            # logging.info('Robust CBBA Computing Time:'+str(cal_time3))

            # routingMethod = 'DRL_planner'
            # actual_reward5 = actualReward(env0, multiObs, routingMethod, DRL_model, CBBA_Assignments3, CBBA_Class3, loaded_model)
            # actualReweards5.append(actual_reward5)

            # logging.info('Robust CBBA Actual Score - DRL_planner:'+str(actual_reward5))
            
        # ActualReweards1.append(np.mean(actualReweards1))
        # ExpectReweards1.append(np.mean(expectReweards1))
        # ActualReweards2.append(np.mean(actualReweards2))
        # ExpectReweards2.append(np.mean(expectReweards2))
        # ActualReweards3.append(np.mean(actualReweards3))
        # ExpectReweards3.append(np.mean(expectReweards3))
        # ActualReweards4.append(np.mean(actualReweards4))
        # ExpectReweards4.append(np.mean(expectReweards4))
        # ActualReweards5.append(np.mean(actualReweards5))
        logging.info('\n')
        logging.info('Average DNN-CBBA Expected Score:'+str(np.mean(expectReweards1)))
        logging.info('Average DNN-CBBA Actual Score:'+str(np.mean(actualReweards1)))
        logging.info('Average Double DNN-CBBA Expected Score:'+str(np.mean(expectReweards_ddnn)))
        logging.info('Average Double DNN-CBBA Actual Score:'+str(np.mean(actualReweards_ddnn)))
        logging.info('Average CBBA Expected Score:'+str(np.mean(expectReweards2)))
        logging.info('Average CBBA Actual Score:'+str(np.mean(actualReweards2)))
        # logging.info('Average Robust CBBA Expected Score:'+str(np.mean(expectReweards3)))
        # logging.info('Average Robust CBBA Actual Score:'+str(np.mean(actualReweards3)))
        logging.info('Average CBBA Actual Score (DRL):'+str(np.mean(actualReweards4)))
        # logging.info('Average Robust CBBA Actual Score (DRL):'+str(np.mean(actualReweards5)))   
        logging.info('Average DNN-CBBA-wo-DRL Expected Score:'+str(np.mean(expectReweards5)))
        logging.info('Average DNN-CBBA-wo-DRL Actual Score:'+str(np.mean(actualReweards5)))
        
        logging.info('\n')
        logging.info('Total delayed time (DNN-CBBA):'+str(np.mean(delayTimes1))) 
        logging.info('Total delayed time (CBBA):'+str(np.mean(delayTimes2)))  
        logging.info('Total delayed time (CBBA-DRL):'+str(np.mean(delayTimes4)))   
        # CalTimes1.append(np.mean(CalTime1))
        # CalTimes2.append(np.mean(CalTime2))
        # CalTimes3.append(np.mean(CalTime3))
    


