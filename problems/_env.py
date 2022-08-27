import torch

class VRP_Environment:
    VEH_STATE_SIZE = 4
    CUST_FEAT_SIZE = 3

    def __init__(self, data, nodes = None, cust_mask = None,
            pending_cost = 2):
        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()
    
        self.pending_cost = pending_cost

    def _update_vehicles(self, dest):
        dist = torch.pairwise_distance(self.cur_veh[:,0,:2], dest[:,0,:2], keepdim = True)
        tt = dist / self.veh_speed

        self.cur_veh[:,:,:2] = dest[:,:,:2]
        self.cur_veh[:,:,2] -= dest[:,:,2]
        self.cur_veh[:,:,3] += tt

        self.vehicles = self.vehicles.scatter(1,
                self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE), self.cur_veh)
        return dist

    def _update_done(self, cust_idx):
        self.veh_done.scatter_(1, self.cur_veh_idx, cust_idx == 0)
        self.done = bool(self.veh_done.all())

    def _update_mask(self, cust_idx):
        self.new_customers = False
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        overload = torch.zeros_like(self.mask).scatter_(1,
                self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count),
                self.cur_veh[:,:,None,2] - self.nodes[:,None,:,2] < 0)
        self.mask = self.mask | self.served[:,None,:] | overload | self.veh_done[:,:,None]
        self.mask[:,:,0] = 0

    def _update_cur_veh(self):
        avail = self.vehicles[:,:,3].clone()
        avail[self.veh_done] = float('inf')
        self.cur_veh_idx = avail.argmin(1, keepdim = True)
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count))

    def reset(self):
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
        self.vehicles[:,:,:2] = self.nodes[:,0:1,:2]
        self.vehicles[:,:,2] = self.veh_capa

        self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype = torch.bool)
        self.done = False

        self.cust_mask = self.init_cust_mask
        self.new_customers = True
        self.served = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype = torch.bool)

        self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype = torch.bool) \
                if self.cust_mask is None else self.cust_mask[:,None,:].repeat(1, self.veh_count, 1)

        self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype = torch.int64)
        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count))

    def step(self, cust_idx):
        dest = self.nodes.gather(1, cust_idx[:,:,None].expand(-1,-1,self.CUST_FEAT_SIZE))
        dist = self._update_vehicles(dest)
        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        self._update_cur_veh()
        reward = -dist
        if self.done:
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
            pending = (self.served ^ True).float().sum(-1, keepdim = True) - 1
            reward -= self.pending_cost * pending
        return reward

    def state_dict(self, dest_dict = None):
        if dest_dict is None:
            dest_dict = {
                    "vehicles": self.vehicles,
                    "veh_done": self.veh_done,
                    "served": self.served,
                    "mask": self.mask,
                    "cur_veh_idx": self.cur_veh_idx
                    }
        else:
            dest_dict["vehicles"].copy_(self.vehicles)
            dest_dict["veh_done"].copy_(self.veh_done)
            dest_dict["served"].copy_(self.served)
            dest_dict["mask"].copy_(self.mask)
            dest_dict["cur_veh_idx"].copy_(self.cur_veh_idx)
        return dest_dict

    def load_state_dict(self, state_dict):
        self.vehicles.copy_(state_dict["vehicles"])
        self.veh_done.copy_(state_dict["veh_done"])
        self.served.copy_(state_dict["served"])
        self.mask.copy_(state_dict["mask"])
        self.cur_veh_idx.copy_(state_dict["cur_veh_idx"])

        self.cur_veh = self.vehicles.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1, self.cur_veh_idx[:,:,None].expand(-1,-1,self.nodes_count))
