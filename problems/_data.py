import torch
import numpy as np
from torch.utils.data import Dataset

class VRP_Dataset(Dataset):
    CUST_FEAT_SIZE = 4

    @classmethod
    def generate(cls,
            batch_size = 1,
            cust_count = 100,
            veh_count = 25,
            veh_capa_range = (50,100),
            veh_speed_range = (0.5,1.0),
            min_cust_count = None,
            cust_loc_range = (0,101),
            cust_dem_range = (5,41),
            cust_rew_range = (0.5,1.0)
            ):
        size = (batch_size, cust_count, 1)

        # Sample locs        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count+1, 2), dtype = torch.float)
        # Sample dems             q_j ~ U(5,  40)
        dems = torch.randint(*cust_dem_range, size, dtype = torch.float)
        # Sample rewards          r_j ~ U(0.5, 1.0)
        rews = torch.randint(*cust_rew_range, size, dtype = torch.float)

        # Regroup all features in one tensor
        customers = torch.cat((locs[:,1:], dems, rews), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:,:,:2] = locs[:,0:1]
        nodes = torch.cat((depot_node, customers), 1)

        if min_cust_count is None:
            cust_mask = None
        else:
            counts = torch.randint(min_cust_count+1, cust_count+2, (batch_size, 1), dtype = torch.int64)
            cust_mask = torch.arange(cust_count+1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0

        # sample veh_capa
        veh_capa = torch.randint(*veh_capa_range, (batch_size, veh_count+1, 2), dtype = torch.float)
        # sample veh_speed
        veh_speed = torch.randint(*veh_speed_range, (batch_size, veh_count+1, 2), dtype = torch.float)

        vehs =  torch.cat((veh_capa, veh_speed), 1)


        dataset = cls(vehs, veh_capa_range, veh_speed_range, nodes, cust_mask)
        return dataset

    def __init__(self, vehs, veh_capa_range, veh_speed_range, nodes, cust_mask = None,):
        self.vehs = vehs
        self.nodes = nodes
        self.veh_capa_range = veh_capa_range
        self.veh_speed_range = veh_speed_range

        self.batch_size, self.nodes_count, d = nodes.size()
        if d != self.CUST_FEAT_SIZE:
            raise ValueError("Expected {} customer features per nodes, got {}".format(
                self.CUST_FEAT_SIZE, d))
        self.cust_mask = cust_mask

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        if self.cust_mask is None:
            return self.nodes[i]
        else:
            return self.nodes[i], self.cust_mask[i]

    def nodes_gen(self):
        if self.cust_mask is None:
            yield from self.nodes
        else:
            yield from (n[m^1] for n,m in zip(self.nodes, self.cust_mask))


    def normalize(self):
        loc_scl, loc_off = self.nodes[:,:,:2].max().item(), self.nodes[:,:,:2].min().item()
        loc_scl -= loc_off

        self.nodes[:,:,:2] -= loc_off
        self.nodes[:,:,:2] /= loc_scl
        self.nodes[:,:, 2] /= self.veh_capa_range[1]

        self.veh_capa /= self.veh_capa_range[1]
        self.veh_speed /= self.veh_speed_range[1]

        return loc_scl, 1

    def save(self, fpath):
        torch.save({
            "vehs": self.vehs,
            "nodes": self.nodes,
            "cust_mask": self.cust_mask
            }, fpath)

    @classmethod
    def load(cls, fpath):
        return cls(**torch.load(fpath))
