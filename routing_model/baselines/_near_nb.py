from routing_model.baselines._base import Baseline

import torch

class NearestNeighbourBaseline(Baseline):
    _BIG_FLOAT = 1e9

    def __init__(self, learner, use_cumul_reward = False):
        super().__init__(learner, use_cumul_reward)
        self.buf = None

    def eval(self, dyna):
        dyna.reset()
        return self.eval_step(dyna, None, None)

    def eval_step(self, dyna, learner_compat, learner_cust_idx):
        self.buf = dyna.state_dict(self.buf)
        rewards = []
        done = False
        while not dyna.done:
            veh_pos = dyna.cur_veh[:,:,:2].unsqueeze(2).expand(-1,-1,dyna.nodes_count,-1)
            cust_pos = dyna.nodes[:,:,:2].unsqueeze(1)
            sqd = (veh_pos - cust_pos).pow(2).sum(dim = 3)
            sqd[:,0,0] += 0.5*self._BIG_FLOAT # Discourage depot unless nothing else possible..
            cust_idx = (sqd + dyna.cur_veh_mask.float() * self._BIG_FLOAT).argmin(dim = 2)
            reward = dyna.step(cust_idx)
            rewards.append(reward)
        dyna.load_state_dict(self.buf)
        return torch.stack(rewards).sum(dim = 0)
