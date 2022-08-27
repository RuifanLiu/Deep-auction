#!/usr/bin/env python3

from marpdan.layers import MultiHeadAttention
import torch

mha = MultiHeadAttention(8, 4, 128)

q = torch.rand(512,2,4)
k = torch.rand(512,11,128)
v = torch.rand(512,11,128)

m = torch.zeros(512,11, dtype = torch.uint8)
m[:,1::2] = 1
h_masked = mha(q,k,v,m)
h_indexed = mha(q,k[:,::2],v[:,::2])

print( "Abs diff between masked and pre-indexed version: max = {}, mean = {}".format(
    (h_masked - h_indexed).abs().max(),
    (h_masked - h_indexed).abs().mean()
    ))

m = torch.zeros(512,11, dtype = torch.uint8)
m[:,1::3] = 1
m[:,2::3] = 1
h_masked = mha(q,k,v,m)
h_indexed = mha(q,k[:,::3],v[:,::3])

print( "Abs diff between masked and pre-indexed version: max = {}, mean = {}".format(
    (h_masked - h_indexed).abs().max(),
    (h_masked - h_indexed).abs().mean()
    ))

for l in range(2, 8):
    m = torch.zeros(512,11, dtype = torch.uint8)
    m[:,l:] = 1
    h_masked = mha(q,k,v,m)
    h_indexed = mha(q,k[:,:l],v[:,:l])

    print( "Abs diff between masked and pre-indexed version: max = {}, mean = {}".format(
        (h_masked - h_indexed).abs().max(),
        (h_masked - h_indexed).abs().mean()
        ))

for l in range(2, 8):
    m = torch.zeros(512,11, dtype = torch.uint8)
    m[:,:l] = 1
    h_masked = mha(q,k,v,m)
    h_indexed = mha(q,k[:,l:],v[:,l:])

    print( "Abs diff between masked and pre-indexed version: max = {}, mean = {}".format(
        (h_masked - h_indexed).abs().max(),
        (h_masked - h_indexed).abs().mean()
        ))
