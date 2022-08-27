# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:57:09 2020

@author: s313488
"""
from stellargraph import StellarGraph
import numpy as np
import pandas as pd

def obs2Graph(obs_list):
    graphOutput = []
    for obs in obs_list:

        nodeFeatureArray = obs.transpose()
        
        # square_weighted_edges = []
        # node_number = np.size(obs,1)
        # for n1 in range(node_number):
        #     for n2 in range(node_number):
        #         edge_feature = {
        #             'source': n1,
        #             'target': n2,
        #             'weight': edgeFeatureArray[n1,n2]}
        #         square_weighted_edges.append(edge_feature)
        # square_weighted_edges = pd.DataFrame(square_weighted_edges)
        
        graphOutput.append(StellarGraph(nodes = nodeFeatureArray))
    return graphOutput
            