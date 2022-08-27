#!/usr/bin/env python3

from marpdan.layers import TransformerEncoderLayer
import torch

enc = TransformerEncoderLayer(8, 128, 512)
enc.eval()

v = torch.rand(512,11,128)

m = torch.zeros(512,11, dtype = torch.uint8)
m[:,1::2] = 1
h_masked = enc(v, mask = m)
h_indexed = enc(v[:,::2])

print( "Abs diff between masked and pre-indexed version: max = {}, mean = {}".format(
    (h_masked[:,::2] - h_indexed).abs().max(),
    (h_masked[:,::2] - h_indexed).abs().mean()
    ))

m = torch.zeros(512,11, dtype = torch.uint8)
m[:,1::3] = 1
m[:,2::3] = 1
h_masked = enc(v,  m)
h_indexed = enc(v[:,::3])

print( "Abs diff between masked and pre-indexed version: max = {}, mean = {}".format(
    (h_masked[:,::3] - h_indexed).abs().max(),
    (h_masked[:,::3] - h_indexed).abs().mean()
    ))

for l in range(2, 8):
    m = torch.zeros(512,11, dtype = torch.uint8)
    m[:,l:] = 1
    h_masked = enc(v, m)
    h_indexed = enc(v[:,:l])

    print( "Abs diff between masked and pre-indexed version: max = {}, mean = {}".format(
        (h_masked[:,:l] - h_indexed).abs().max(),
        (h_masked[:,:l] - h_indexed).abs().mean()
        ))

for l in range(2, 8):
    m = torch.zeros(512,11, dtype = torch.uint8)
    m[:,:l] = 1
    h_masked = enc(v, m)
    h_indexed = enc(v[:,l:])

    print( "Abs diff between masked and pre-indexed version: max = {}, mean = {}".format(
        (h_masked[:,l:] - h_indexed).abs().max(),
        (h_masked[:,l:] - h_indexed).abs().mean()
        ))
