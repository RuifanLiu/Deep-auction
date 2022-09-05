
import os

import itertools
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

# from utils.GraphTransfer import obs2Graph
import stellargraph as sg
# from stellargraph.mapper import PaddedGraphGenerator
# from stellargraph.layer import DeepGraphCNN #, GCNSupervisedGraphClassification, GAT, GATSupervisedGraphClassification

from sklearn import datasets, model_selection
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import Model, regularizers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten, GlobalMaxPooling1D
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error

# from utils.data_util import load_dataset
from utils.networks import MultiHeadAttention
from utils.data_util import data_padding

from customPolicy.common_policies import A2C_ValueNetwork

from customEnv.costum_vec_env import VecNormalize
from customEnv.SubEnvironment import TAEnv_submodular
from customEnv.costum_vec_env import make_ta_env

tf.enable_eager_execution()

def normalize_fixed(obses, values):

    obs_min = np.expand_dims(np.array([0.5, 0.75, 0,  0, 0]), axis=-1)
    obs_max = np.expand_dims(np.array([1.0, 1.25, 20,  1, 1]), axis=-1)

    obses_normed = [(obs-obs_min)/(obs_max-obs_min) for obs in obses]

    # value_min = np.min(values)
    # value_max = np.max(values)
    value_min, value_max = 0.0, 10.0
    values_normed = [(value-value_min)/(value_max-value_min) for value in values]
    return obses_normed, values_normed

def make_dataset(filepath):
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)


    data_features = []
    len_ds = 0
    for data in dataset:
        data_feat = data['Input_Obs'][0]
        data_feat_padded = data_padding(data_feat, pad_size=20)
        data_features.append(data_feat_padded)
        len_ds += 1
    data_features = tf.stack(data_features)
    data_values = [np.array([data['input_Value']]) for data in dataset]
    data_values = tf.stack(data_values)

    dataset = tf.data.Dataset.from_tensor_slices((data_features, data_values))

    # len_ds = len(dataset)
    # data_features = [data['Input_Obs'][0] for data in dataset]
    # data_values = [np.array([data['input_Value']]) for data in dataset]

    # # data_features, data_values = normalize_fixed(data_features, data_values)

    # dataset = tf.data.Dataset.from_generator(lambda: 
    #                           itertools.zip_longest(data_features, data_values),
    #                           output_types=(tf.float32, tf.float32),
    #                           output_shapes=(tf.TensorShape([5, None]), tf.TensorShape([1,])))

    return dataset, len_ds



if __name__ == '__main__':

    # filepath = 'Simu_result/A2C-2022-06-16-11-18-58/Train_DataSet10000_2022-06-29-22-43-50.pkl'
    filepath = 'Simu_result/A2C-2022-06-16-11-18-58/Train_DataSet10000_2022-06-29-22-43-50.pkl'
    
    dataset, len_ds = make_dataset(filepath)
    batch_size = 64.0
    dataset = dataset.shuffle(buffer_size=len_ds, reshuffle_each_iteration=True)
    train_dataset = dataset.take(int(len_ds*0.8)).batch(batch_size=int(batch_size), drop_remainder=True)
    test_dataset = dataset.skip(int(len_ds*0.8)).batch(batch_size=int(batch_size), drop_remainder=True)

    # train_dataset = dataset.take(int(len_ds*0.8))
    # valid_dataset = train_dataset.take(int(len_ds*0.8*0.2))
    # test_dataset = dataset.skip(int(len_ds*0.8))

    # batch_size = 64
    # train_dataset = train_dataset.padded_batch(batch_size=batch_size,
    #     padded_shapes=(tf.TensorShape([5, None]), tf.TensorShape([1,])))   
    # valid_dataset = valid_dataset.padded_batch(batch_size=batch_size,
    #     padded_shapes=(tf.TensorShape([5, None]), tf.TensorShape([1,])))   
    # test_dataset = test_dataset.padded_batch(batch_size=batch_size,
    #     padded_shapes=(tf.TensorShape([5, None]), tf.TensorShape([1,])))  
    # # dataset = dataset.batch(1)

    # train_dataset = train_dataset.repeat()
    # valid_dataset = valid_dataset.repeat()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6 # avoid out of memory

    dt_string = 'A2C-2022-06-16-11-18-58'
    last_trained_model='Model-A2C-2022-06-16-11-18-58SS-GATPolicy_sub-DOP10'
    RL_algorithm = 'A2C_ValueNetwork'
    policy = 'GATPolicy_sub'
    initial_model = eval(RL_algorithm).load(last_trained_model)

    env0 = TAEnv_submodular(mission_size = 20, mission_stochastic_flag='stochastic', 
                        task_duration_stochastic = 'stochastic', time_discount = 0.9,
                        seed=None, maneuver_target=False, dynamic_task = False)

    env1 = make_ta_env(env0, num_env=1, seed=0, use_subprocess=True)
    env1 = VecNormalize(env1, norm_obs=False, norm_reward=False)

    model = eval(RL_algorithm)(policy=policy, env=env1, batch_size=batch_size, learning_rate=1e-4)
    par = initial_model.get_parameters()
    model.load_parameters(par)

    now = datetime.now() #analyse the time consumption
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S") #time string

    model.learn(train_dataset, test_dataset, total_epoch = 50, tb_log_name='log-valueNetworkTrain-'+dt_string) 
    
    # Save the agent  
    ModelFilename = 'ValueNetwork-'+dt_string
    SimuSavePath = './valueNN_model'
    model_path = os.path.join(SimuSavePath, ModelFilename)
    model.save(model_path)