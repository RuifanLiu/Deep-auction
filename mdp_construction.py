import torch
from torch.utils.data import DataLoader

import os
import pickle
from tqdm import tqdm

from cbba._mdp_utils import MDP
from mdptoolbox.mdp import PolicyIteration, FiniteHorizon, ValueIteration

if __name__ == '__main__':

    for n in range(2,11):
        m = 2
        data_path = "./data_sample10_stw/s_cvrptw_n{}m{}/norm_data.pyth".format(n, m)    
        data = torch.load(data_path)
        loader = DataLoader(data, batch_size = 1)
        for i, batch in enumerate(tqdm(loader)):
            out_dir = "mdp_sample10_stw/s_cvrptw_n{}m{}".format(n,m)
            os.makedirs(out_dir, exist_ok = True)
            save_path = os.path.join(out_dir, 'mdp{}.pkl'.format(i))
            if not os.path.exists(save_path):
                print('constucting the mdp for ' + save_path + '...')
                batch_mdp = MDP(batch)
                tp_mtx, rew_mtx= batch_mdp.mdp_generation()
                pi = PolicyIteration(tp_mtx, rew_mtx, discount=0.99)
                pi.run()
                with open(save_path, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump([batch_mdp, pi], f, pickle.HIGHEST_PROTOCOL)
            else:
                print('skip the mdp for ' + save_path)
