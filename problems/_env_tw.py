from problems import VRP_Environment
import torch

class VRPTW_Environment(VRP_Environment):
    VEH_STATE_SIZE = 5 # [loc, cap, speed, time]
    CUST_FEAT_SIZE = 7 # [loc, dem, rew, rdys, ends, durs]

    def __init__(self, data, vehs = None, nodes = None, cust_mask = None,
            pending_cost = 2, late_cost = 1):
        super().__init__(data, vehs, nodes, cust_mask, pending_cost)
        # self.late_discount = late_discount
        self.late_cost = late_cost

    def _sample_speed(self):
        return self.cur_veh[:,:,3]

    def _update_vehicles(self, dest):
        dist = torch.pairwise_distance(self.cur_veh[:,0,:2], dest[:,0,:2], keepdim = True)
        tt = dist / self._sample_speed()
        arv = torch.max(self.cur_veh[:,:,4] + tt, dest[:,:,4])
        late = ( arv - dest[:,:,5] ).clamp_(min = 0)
        self.cur_veh[:,:,:2] = dest[:,:,:2]
        self.cur_veh[:,:,2] -= dest[:,:,2]
        self.cur_veh[:,:,4] = arv + dest[:,:,6]

        # finish = torch.le(self.cur_veh[:,:,4],dest[:,:,5])
        # reward = finish * dest[:,:,3] * torch.exp(-self.late_discount*late)

        self.vehicles = self.vehicles.scatter(1,
                self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE), self.cur_veh)
        return dist, late

    def step(self, cust_idx):
        dest = self.nodes.gather(1, cust_idx[:,:,None].expand(-1,-1,self.CUST_FEAT_SIZE))
        dist, late = self._update_vehicles(dest)
        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        self._update_cur_veh()
        # reward = reward
        reward = -dist - self.late_cost * late
        if self.done:
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
            pending = (self.served ^ True).float().sum(-1, keepdim = True) - 1
            reward -= self.pending_cost * pending
        return reward
