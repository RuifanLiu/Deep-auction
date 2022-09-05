import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


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

def _padding_with_zeros(custs, pad_size):
    batch_size, nodes_count, d = custs.size()
    padded_custs = torch.nn.functional.pad(custs, (0, 0, 0, pad_size-nodes_count, 0, 0),\
        mode='constant', value=0)
    return padded_custs


class PaddedData(Dataset):

    def __init__(self, vehs, nodes, padding_size=None):
        self.vehs = vehs

        self.batch_size, self.cust_count, d = nodes.size()
        self.padding_size = padding_size if padding_size is not None else self.cust_count
        if self.cust_count > self.padding_size:
            raise ValueError('the paddding size shall be larger than data')
        
        self._padded_nodes = torch.zeros(self.batch_size, self.padding_size, d)
        self._padded_nodes[:,:self.cust_count,:] = nodes
        self.nodes = self._padded_nodes
        self.cust_mask = torch.arange(self.padding_size).expand(self.batch_size,-1) > (self.cust_count - 1)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        if self.cust_mask is None:
            return self.vehs[i], self.nodes[i]
        else:
            return self.vehs[i], self.nodes[i], self.cust_mask[i]

    def nodes_gen(self):
        if self.cust_mask is None:
            yield from self.nodes
        else:
            yield from (n[m^1] for n,m in zip(self.nodes, self.cust_mask))
