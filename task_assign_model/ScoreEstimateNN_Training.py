# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:16:17 2020

@author: s313488
"""
import os
import itertools
import logging
import math
from re import X
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
from tensorflow.keras import backend as K

# from utils.data_util import load_dataset
from utils.networks import MultiHeadAttention
from utils.data_util import data_padding

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


    # data_features = []
    # len_ds = 0
    # for data in dataset:
    #     data_feat = data['Input_Obs'][0]
    #     data_feat_padded = data_padding(data_feat, pad_size=20)
    #     data_features.append(data_feat_padded)
    #     len_ds += 1
    # data_features = tf.stack(data_features)
    # data_values = [np.array([data['input_Value']]) for data in dataset]
    # data_values = tf.stack(data_values)

    # dataset = tf.data.Dataset.from_tensor_slices((data_features, data_values))

    len_ds = len(dataset)
    data_features = [data['Input_Obs'] for data in dataset]
    data_values = [np.array([data['input_Value']]) for data in dataset]

    # data_features, data_values = normalize_fixed(data_features, data_values)

    dataset = tf.data.Dataset.from_generator(lambda: 
                              itertools.zip_longest(data_features, data_values),
                              output_types=(tf.float32, tf.float32),
                              output_shapes=(tf.TensorShape([5, None]), tf.TensorShape([1,])))

    return dataset, len_ds

def creat_model(input_shape, units, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0,):
    ### Transformer ######
    # Attention and Normalization


    def transformer_encoder(inputs, units, num_heads, ff_dim, dropout, mask):
        x, attn = MultiHeadAttention(d_model=units, num_heads=num_heads)(inputs, k=inputs, q=inputs, mask=mask)
        # x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Dense(ff_dim, activation="relu")(res)  # (batch_size, graph_size, dims)
        # x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(ff_dim, activation='relu')(x)
        # x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        # x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        return res + x
    
    inputs = keras.Input(shape=input_shape)

    x = tf.transpose(inputs, perm=(0, 2, 1)) 
    x = layers.Dense(units=units, activation='relu')(x)

    # fir_row = inputs[:, 0, :]
    # fir_row_binary = 1-tf.cast(fir_row>0, tf.float32)# tf.where(fir_row, tf.zeros_like(fir_row), tf.ones_like(fir_row)) # (-1,node_size)
    # mask1 = tf.expand_dims(tf.expand_dims(fir_row_binary, 1),-1)
    # mask2 = tf.expand_dims(tf.expand_dims(fir_row_binary, 1),1)
    # mask = tf.matmul(mask1, mask2)

    mask = None

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, units, num_heads, ff_dim, dropout, mask) # (batch_size, graph_size, dims)
    
    # # x = layers.GlobalAveragePooling1D(data_format="channels_first")(x) # (batch_size, dims)

    # ### Add decoder to the network
    # task_query = inputs[:,:,0] # extract the first task
    # task_query = layers.Dense(units=units, activation='relu')(task_query)
    # glimpse_K = layers.Dense(units=units, activation=None, use_bias=False)(x)
    # glimpse_V = layers.Dense(units=units, activation=None, use_bias=False)(x)
    # logit_K = layers.Dense(units=units, activation=None, use_bias=False)(x)

    # glimpse_Q  = tf.reshape(task_query,[-1, 1, units])

    # compatibility = tf.matmul(glimpse_Q, tf.transpose(glimpse_K, perm=(0, 2, 1))) / math.sqrt(units)
    # glimpse = tf.matmul(layers.Softmax(axis=-1)(compatibility), glimpse_V)

    x = tf.reduce_mean(x, axis=1)

    for dim in mlp_units:
        project_out = layers.Dense(units=dim, activation="relu")(x)
        project_out = layers.Dropout(mlp_dropout)(project_out)

    outputs = layers.Dense(1, activation=None)(project_out)

    return keras.Model(inputs, outputs)

if __name__ == '__main__':

    # filepath = 'Simu_result/A2C-2022-06-16-11-18-58/Train_DataSet10000_2022-06-29-22-43-50.pkl'
    datesetPath = 'Simu_result/A2C-2022-06-16-11-18-58/Train_DataSet-task100-size10000_2022-07-19-22-13-42.pkl'
    now = datetime.now() #analyse the time consumption
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S") #time string

    # Logging
    file_path = './Saved_model'
    file_name = 'log-'+dt_string+'.txt'

    file_name = os.path.join(file_path, file_name)
    logging.basicConfig(filename=file_name,
                level=logging.INFO,
                format='%(message)s',  #%(levelname)s:
                datefmt='%I:%M:%S')

    logging.info('DataSet:'+ datesetPath)
    
    dataset, len_ds = make_dataset(datesetPath)
    dataset = dataset.shuffle(buffer_size=len_ds, reshuffle_each_iteration=True)
    # train_dataset = dataset.take(int(len_ds*0.8)).batch(batch_size=64)
    # test_dataset = dataset.skip(int(len_ds*0.8)).batch(batch_size=64)

    train_dataset = dataset.take(int(len_ds*0.8))
    valid_dataset = train_dataset.take(int(len_ds*0.8*0.2))
    test_dataset = dataset.skip(int(len_ds*0.8))

    batch_size = 64
    padded_size = 100
    train_dataset = train_dataset.padded_batch(batch_size=batch_size,
        padded_shapes=(tf.TensorShape([5, padded_size]), tf.TensorShape([1,])))   
    valid_dataset = valid_dataset.padded_batch(batch_size=batch_size,
        padded_shapes=(tf.TensorShape([5, padded_size]), tf.TensorShape([1,])))   
    test_dataset = test_dataset.padded_batch(batch_size=batch_size,
        padded_shapes=(tf.TensorShape([5, padded_size]), tf.TensorShape([1,])))  
    # dataset = dataset.batch(1)

    train_dataset = train_dataset.repeat()
    valid_dataset = valid_dataset.repeat()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6 # avoid out of memory

    input_shape = (5, None)
    model = creat_model(input_shape, units=64, num_heads=4, ff_dim=64,
                        num_transformer_blocks=3, 
                        mlp_units=[128, 128], mlp_dropout=0.0, dropout=0.0)

    model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error', metrics=["mse"])

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
 
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=100, \
        verbose=1, shuffle=True, steps_per_epoch=int(len_ds*0.8/batch_size), validation_steps=int(len_ds*0.8*0.2/batch_size))
   
    _, rmse = model.evaluate(test_dataset, verbose=1, steps=int(len_ds*0.2/batch_size))

    # save the fitted model and the fitting history
    modelSavePath = "./Saved_model/my_model-"+dt_string
    model.save(modelSavePath)
    historySavePath = "./Saved_model/history-"+dt_string
    with open(historySavePath, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # history = pickle.load(open(modelSavePath), "rb")

    for epoch in history.epoch:
        mse = history.history['mean_squared_error'][epoch]
        val_mse = history.history['val_mean_squared_error'][epoch]
        logging.info(f'Epoch {epoch} : mean_squared_error: {mse} \t val_mean_squared_error:  {val_mse}')

    # Training figure save
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model loss')
    plt.ylabel('mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    figsavepath = './Saved_model/TrainingProcess-'+dt_string
    plt.savefig(figsavepath+'.png', dpi=300)

    test_metrics = model.evaluate(test_dataset, steps=1000)
    print("\nTest Set Metrics:")
    logging.info("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
        logging.info("\t{}: {:0.4f}".format(name, val))
    


    