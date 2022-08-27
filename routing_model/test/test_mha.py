#!/usr/bin/env python3

from marpdan.layers._mha import _MHA_V1, _MHA_V2

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    import time
    from itertools import chain

    mha = _MHA_V1(8,4,128)
    mha_v2 = _MHA_V2(8,4,128)
    mha_v2.load_state_dict(mha.state_dict())
    proj = nn.Linear(128,1)

    q = torch.rand(512,1,4)
    k = torch.rand(512,10,128)
    v = torch.rand(512,10,128)
    m = torch.randint(0,2, (512,1,10), dtype = torch.uint8)
    gt = torch.zeros(512,1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mha.to(dev)
    mha_v2.to(dev)
    proj.to(dev)
    q,k,v,m,gt = q.to(dev), k.to(dev), v.to(dev), m.to(dev), gt.to(dev)

    IT = 100

    ref = mha(q,k,v,m)
    x = proj(ref.squeeze(1))
    loss = F.smooth_l1_loss(x, gt)
    loss.backward()
    grad_ref = [p.grad for p in mha.parameters()]
    all_close_fw = 0
    all_close_bw = 0

    for i,mha_ver in enumerate([mha, mha_v2]):
        mu_t_fw = 0
        mu_t_bw = 0
        for it in range(IT):
            st_t = time.monotonic()
            o = mha_ver(q,k,v,m)
            mu_t_fw += (time.monotonic() - st_t) / IT
            if torch.allclose(o,ref, atol = 1e-9):
                all_close_fw += 1

            x = proj(o.squeeze(1))
            loss = F.smooth_l1_loss(x, gt)
            mha_ver.zero_grad()
            proj.zero_grad()
            st_t = time.monotonic()
            loss.backward()
            mu_t_bw += (time.monotonic() - st_t) / IT
            if all( torch.allclose(p.grad, gp_ref, atol = 1e-9) for p,gp_ref in zip(mha_ver.parameters(), grad_ref) ):
                all_close_bw += 1
        print("V{} : \t\t FW = {:.3f}ms match {:.0%} \t\t BW = {:.3f}ms match {:.0%}".format(i+1, mu_t_fw * 1000, all_close_fw / IT, mu_t_bw * 1000, all_close_bw / IT))
