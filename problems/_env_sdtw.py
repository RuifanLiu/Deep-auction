from problems import SVRPTW_Environment
import torch

class SDVRPTW_Environment(SVRPTW_Environment):
    CUST_FEAT_SIZE = 7

    def _update_hidden(self):
        time = self.cur_veh[:, :, 3].clone()
        if self.init_cust_mask is None:
            reveal = self.cust_mask & (self.nodes[:,:,6] <= time)
        else:
            reveal = (self.cust_mask ^ self.init_cust_mask) & (self.nodes[:,:,6] < time)
        if reveal.any():
            self.new_customers = True
            self.cust_mask = self.cust_mask ^ reveal
            self.mask = self.mask ^ reveal[:,None,:].expand(-1,self.veh_count,-1)
            self.veh_done = self.veh_done & (reveal.any(1) ^ True).unsqueeze(1)
            self.vehicles[:, :, 3] = torch.max(self.vehicles[:, :, 3], time)
            self._update_cur_veh()

    def reset(self):
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
        self.vehicles[:,:,:2] = self.nodes[:,0:1,:2]
        self.vehicles[:,:,2] = self.veh_capa

        self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype = torch.bool)
        self.done = False

        self.cust_mask = (self.nodes[:,:,6] > 0)
        if self.init_cust_mask is not None:
            self.cust_mask = self.cust_mask | self.init_cust_mask
        self.new_customers = True
        self.served = torch.zeros_like(self.cust_mask)

        self.mask = self.cust_mask[:,None,:].repeat(1, self.veh_count, 1)

        self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype = torch.int64)
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count))

    def step(self, cust_idx):
        reward = super().step(cust_idx)
        self._update_hidden()
        return reward
