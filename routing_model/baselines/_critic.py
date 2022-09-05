from routing_model.baselines._base import Baseline

import torch
import torch.nn as nn


class CriticBaseline(Baseline):
    def __init__(self, learner, cust_count, use_qval = True, use_cumul_reward = False):
        super().__init__(learner, use_cumul_reward)
        self.use_qval = use_qval
        self.cust_count = cust_count
        self.project = nn.Linear(cust_count+1, cust_count+1 if use_qval else 1, bias = False)

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx):
        compat = learner_compat.clone()
        compat[vrp_dynamics.cur_veh_mask] = 0
        val = self.project(compat)
        if self.use_qval:
            val = val.gather(2, cust_idx.unsqueeze(1).expand(-1,1,-1))
        return val.squeeze(1)
    
    def eval_init(self, vrp_dynamics):
        self.learner._encode_customers(vrp_dynamics.nodes)
        vrp_dynamics.reset()
        veh_repr = self.learner._repr_vehicle(
                    vrp_dynamics.vehicles,
                    vrp_dynamics.cur_veh_idx,
                    vrp_dynamics.mask)
        compat = self.learner._score_customers(veh_repr)
        compat[vrp_dynamics.cur_veh_mask] = 0
        val = self.project(compat)
        return val.squeeze(1)

    def __call__(self, vrp_dynamics):
        self.learner._encode_customers(vrp_dynamics.nodes)
        vrp_dynamics.reset()
        actions, logps, rewards, bl_vals = [], [], [], []
        while not vrp_dynamics.done:
            veh_repr = self.learner._repr_vehicle(
                    vrp_dynamics.vehicles,
                    vrp_dynamics.cur_veh_idx,
                    vrp_dynamics.mask)
            compat = self.learner._score_customers(veh_repr)
            logp = self.learner._get_logp(compat, vrp_dynamics.cur_veh_mask)
            cust_idx = logp.exp().multinomial(1)
            if not(self.use_cumul and bl_vals):
                bl_vals.append( self.eval_step(vrp_dynamics, compat, cust_idx) )
            actions.append( (vrp_dynamics.cur_veh_idx, cust_idx) )
            logps.append( logp.gather(1, cust_idx) )
            r = vrp_dynamics.step(cust_idx)
            rewards.append(r)
        if self.use_cumul:
            rewards = torch.stack(rewards).sum(dim = 0)
            bl_vals = bl_vals[0]
        return actions, logps, rewards, bl_vals

    def parameters(self):
        return self.project.parameters()

    def state_dict(self):
        return self.project.state_dict()

    def load_state_dict(self, state_dict):
        return self.project.load_state_dict(state_dict)

    def to(self, device):
        self.project.to(device = device)
