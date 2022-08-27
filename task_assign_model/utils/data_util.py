import os
import pickle
import numpy as np


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)

def data_padding(data, pad_size):
    num_node = np.shape(data)[1]

    element = np.array([0, 0, 0, 1, 0])
    padded_part = np.tile(element,[pad_size-num_node,1])
    padded_part_trans = np.transpose(padded_part)

    data_padded = np.concatenate((data,padded_part_trans), axis=1)
    return data_padded

def data_padding_zeros(data, pad_size):
    num_node = np.shape(data)[1]

    element = np.array([0, 0, 0, 0, 0])
    padded_part = np.tile(element,[pad_size-num_node,1])
    padded_part_trans = np.transpose(padded_part)

    data_padded = np.concatenate((data,padded_part_trans), axis=1)
    return data_padded