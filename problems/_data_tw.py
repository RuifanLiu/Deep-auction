from problems import VRP_Dataset
import torch

class VRPTW_Dataset(VRP_Dataset):
    CUST_FEAT_SIZE = 7

    @classmethod
    def generate(cls,
            batch_size = 1,
            cust_count = 100,
            veh_count = 25,
            veh_capa_range = (100,100),
            veh_speed_range = (1,3),
            min_cust_count = None,
            max_cust_count = None,
            cust_loc_range = (0,101), 
            cust_dem_range = (0,0),
            cust_rew_range = (1,1),
            horizon = 480,
            cust_dur_range = (10,31),
            tw_ratio = (0.25,0.5,0.75,1.0),
            cust_tw_range = (30,91)
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
        # Sample serv. time       s_j ~ U(10min, 30min)
        durs = torch.randint(*cust_dur_range, size, dtype = torch.float) \
            if cust_dur_range[0]!=cust_dur_range[1] else torch.full(size, cust_dur_range[0], dtype = torch.float)

        # Sample TW subset            ~ B(tw_ratio)
        if isinstance(tw_ratio, float):
            has_tw = torch.empty(size).bernoulli_(tw_ratio)
        elif len(tw_ratio) == 1:
            has_tw = torch.empty(size).bernoulli_(tw_ratio[0])
        else: # tuple of float
            ratio = torch.tensor(tw_ratio)[torch.randint(0, len(tw_ratio), (batch_size,), dtype = torch.int64)]
            has_tw = ratio[:,None,None].expand(*size).bernoulli()

        # Sample TW width        tw_j = H if not in TW subset
        #                        tw_j ~ U(30,90) if in TW subset
        tws = (1 - has_tw) * torch.full(size, horizon) \
                + has_tw * torch.randint(*cust_tw_range, size, dtype = torch.float)

        tts = (locs[:,None,0:1,:] - locs[:,1:,None,:]).pow(2).sum(-1).pow(0.5) / veh_speed_range[0]
        # Sample ready time       e_j = 0 if not in TW subset
        #                         e_j ~ U(a_j, H - max(tt_0j + s_j, tw_j))
        rdys = has_tw * (torch.rand(size) * (horizon - torch.max(tts + durs, tws)))
        rdys.floor_()

        # Regroup all features in one tensor
        customers = torch.cat((locs[:,1:], dems, rews, rdys, rdys + tws, durs), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:,:,:2] = locs[:,0:1]
        depot_node[:,:,5] = horizon
        nodes = torch.cat((depot_node, customers), 1)

        if min_cust_count is None:
            cust_mask = None
        else:
            if max_cust_count is None:
                raise ValueError('missed the max customer count')
            else:
                counts = torch.randint(min_cust_count+1, max_cust_count+2, (batch_size, 1), dtype = torch.int64)
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

        t_scl = horizon
        loc_scl = cust_loc_range[1] - cust_loc_range[0] - 1
        cap_scl = veh_capa_range[1]
        dataset = cls(vehs, nodes, cust_mask, t_scl = t_scl, loc_scl = loc_scl, cap_scl = cap_scl)
        return dataset

    def normalize(self):
        loc_scl = self.loc_scl
        cap_scl = self.cap_scl
        t_scl = self.t_scl

        # loc_scl, loc_off = self.nodes[:,:,:2].max().item(), self.nodes[:,:,:2].min().item()
        # loc_scl -= loc_off
        # cap_scl = self.vehs[:,:,0].max().item() 
        # t_scl = self.nodes[:,:,4].max().item() 
        # self.nodes[:,:,:2] -= loc_off
        self.nodes[:,:,:2] /= loc_scl
        self.nodes[:,:, 2] /= cap_scl
        self.nodes[:,:,4:] /= t_scl

        self.vehs[:,:,0] /= cap_scl
        self.vehs[:,:,1] *= t_scl / loc_scl # 480/100

        return loc_scl, t_scl
