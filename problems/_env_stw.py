from problems import VRPTW_Environment
import torch

class SVRPTW_Environment(VRPTW_Environment):
    def __init__(self, data, vehs = None, nodes = None, cust_mask = None,
            pending_cost = 2,  late_cost = 1,
            speed_var = 0.2, late_p = 0.05, slow_down = 0.5, late_var = 0.3):
        super().__init__(data, vehs, nodes, cust_mask, pending_cost, late_cost)
        self.speed_var = speed_var
        self.late_p = late_p
        self.slow_down = slow_down
        self.late_var = late_var

    def _sample_speed(self):
        late = self.nodes.new_empty((self.minibatch_size, 1)).bernoulli_(self.late_p)
        rand = torch.randn_like(late)
        rand = torch.clip(rand, min=-0.5, max=0.5)
        speed = late * self.slow_down * (1 + self.late_var * rand) + (1-late) * (1 + self.speed_var * rand)
        # speed = 1 + self.speed_var * rand
        # print(speed)
        return speed.clamp_(min = 0.1) * self.cur_veh[:,:,3]

