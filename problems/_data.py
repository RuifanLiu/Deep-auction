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
            veh_capa_range = (100,100),
            veh_speed_range = (1,3),
            min_cust_count = None,
            cust_loc_range = (0,101),
            cust_dem_range = (0,0),
            cust_rew_range = (1,1)
            ):
        size = (batch_size, cust_count, 1)

        # Sample locs        x_j, y_j ~ U(0km, 10km)
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count+1, 2), dtype = torch.float)
        # Sample dems             q_j ~ U(5,  40)
        dems = torch.randint(*cust_dem_range, size, dtype = torch.float) \
            if cust_dem_range[0]!=cust_dem_range[1] else torch.full(size, cust_dem_range[0], dtype = torch.float)
        # Sample rewards          r_j ~ U(0.5, 1.0)
        rews = torch.randint(*cust_rew_range, size, dtype = torch.float) \
            if cust_rew_range[0]!=cust_rew_range[1] else torch.full(size, cust_rew_range[0], dtype = torch.float)


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

        veh_size = (batch_size, veh_count, 1)
        # sample veh_capa    c_i ~ U(50,100)
        veh_capa = torch.randint(*veh_capa_range, veh_size, dtype = torch.float)\
            if veh_capa_range[0]!=veh_capa_range[1] else torch.full(veh_size, veh_capa_range[0], dtype = torch.float)
        # sample veh_speed      v_i ~ U(0.6km/min, 1.2km/min) = (10m/s, 20m/s)
        veh_speed = torch.randint(*veh_speed_range, veh_size, dtype = torch.float) \
            if veh_speed_range[0]!=veh_speed_range[1] else torch.full(veh_size, veh_speed_range[0], dtype = torch.float)
        vehs =  torch.cat((veh_capa, veh_speed), 2)

    
        loc_scl = cust_loc_range[1] - cust_loc_range[0] - 1
        cap_scl = veh_capa_range[1]
        dataset = cls(vehs, nodes, cust_mask, loc_scl = loc_scl, cap_scl = cap_scl)
    
        return dataset

    def __init__(self, vehs, nodes, cust_mask = None, **kwargs):
        self.vehs = vehs
        self.nodes = nodes

        self.batch_size, self.nodes_count, d = nodes.size()
        if d != self.CUST_FEAT_SIZE:
            raise ValueError("Expected {} customer features per nodes, got {}".format(
                self.CUST_FEAT_SIZE, d))
        self.cust_mask = cust_mask

        self.loc_scl = kwargs.get('loc_scl') if 'loc_scl' in kwargs else None       
        self.cap_scl = kwargs.get('cap_scl') if 'cap_scl' in kwargs else None
        self.t_scl = kwargs.get('t_scl') if 't_scl' in kwargs else None

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        if self.cust_mask is None:
            return self.vehs[i], self.nodes[i]
        else:
            return self.vehs[i], self.nodes[i], self.cust_mask[i]

    def nodes_gen(self):
        if self.cust_mask is None:
            yield from self.nodes
        else:
            yield from (n[m^1] for n,m in zip(self.nodes, self.cust_mask))


    def normalize(self):
        # loc_scl, loc_off = self.nodes[:,:,:2].max().item(), self.nodes[:,:,:2].min().item()
        # loc_scl -= loc_off

        # cap_scl = self.vehs[:,:,0].max().item()

        # self.nodes[:,:,:2] -= loc_off
        self.nodes[:,:,:2] /= self.loc_scl
        self.nodes[:,:, 2] /= self.cap_scl

        self.vehs[:,:,0] /= self.cap_scl
        self.vehs[:,:,1] /= self.loc_scl

        return self.loc_scl, self.cap_scl

    def save(self, fpath):
        torch.save({
            "vehs": self.vehs,
            "nodes": self.nodes,
            "cust_mask": self.cust_mask
            }, fpath)

    @classmethod
    def load(cls, fpath):
        return cls(**torch.load(fpath))
