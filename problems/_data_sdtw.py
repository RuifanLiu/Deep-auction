from problems import VRPTW_Dataset
import torch

class SDVRPTW_Dataset(VRPTW_Dataset):
    CUST_FEAT_SIZE = 7

    @classmethod
    def generate(cls,
            batch_size = 1,
            cust_count = 100,
            veh_count = 25,
            veh_capa = 200,
            veh_speed = 1,
            min_cust_count = None,
            cust_loc_range = (0,101),
            cust_dem_range = (5,41),
            horizon = 480,
            cust_dur_range = (10,31),
            tw_ratio = 0.5,
            cust_tw_range = (30,91),
            dod = 0.5,
            d_early_ratio = 0.5
            ):
        size = (batch_size, cust_count, 1)

        # Sample locs        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count+1, 2), dtype = torch.float)
        # Sample dems             q_j ~ U(5,  40)
        dems = torch.randint(*cust_dem_range, size, dtype = torch.float)
        # Sample serv. time       s_j ~ U(10, 30)
        durs = torch.randint(*cust_dur_range, size, dtype = torch.float)

        # Sample dyn subset           ~ B(dod)
        # and early/late appearance   ~ B(d_early_ratio)
        if isinstance(dod, float):
            is_dyn = torch.empty(size).bernoulli_(dod)
        elif len(dod) == 1:
            is_dyn = torch.empty(size).bernoulli_(dod[0])
        else: # tuple of float
            ratio = torch.tensor(dod)[torch.randint(0, len(dod), (batch_size,), dtype = torch.int64)]
            is_dyn = ratio[:,None,None].expand(*size).bernoulli()

        if isinstance(d_early_ratio, float):
            is_dyn_e = torch.empty(size).bernoulli_(d_early_ratio)
        elif len(d_early_ratio) == 1:
            is_dyn_e = torch.empty(size).bernoulli_(d_early_ratio[0])
        else:
            ratio = torch.tensor(d_early_ratio)[
                    torch.randint(0, len(d_early_ratio), (batch_size,), dtype = torch.int64)
                    ]
            is_dyn_e = ratio[:,None,None].expand(*size).bernoulli()

        # Sample appear. time     a_j = 0 if not in D subset
        #                         a_j ~ U(1,H/3) if early appear
        #                         a_j ~ U(H/3+1, 2H/3) if late appear
        aprs = is_dyn * is_dyn_e * torch.randint(1, horizon//3+1, size, dtype = torch.float) \
                + is_dyn * (1-is_dyn_e) * torch.randint(horizon//3+1, 2*horizon//3+1, size, dtype = torch.float)

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

        tts = (locs[:,None,0:1,:] - locs[:,1:,None,:]).pow(2).sum(-1).pow(0.5) / veh_speed
        # Sample ready time       e_j = 0 if not in TW subset
        #                         e_j ~ U(a_j, H - max(tt_0j + s_j, tw_j))
        rdys = has_tw * (aprs + torch.rand(size) * (horizon - torch.max(tts + durs, tws) - aprs))
        rdys.floor_()

        # Regroup all features in one tensor
        customers = torch.cat((locs[:,1:], dems, rdys, rdys + tws, durs, aprs), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:,:,:2] = locs[:,0:1]
        depot_node[:,:,4] = horizon
        nodes = torch.cat((depot_node, customers), 1)

        if min_cust_count is None:
            cust_mask = None
        else:
            counts = torch.randint(min_cust_count+1, cust_count+2, (batch_size, 1), dtype = torch.int64)
            cust_mask = torch.arange(cust_count+1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        return dataset
