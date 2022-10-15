# test the submodularity level of a reward function
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def CBBA_InsertInList(oldList, value, index):
    index = int(index)
    newList = np.copy(oldList)
    newList[:index] = oldList[:index]
    newList[index]       = value
    newList[index+1:] = oldList[index:-1]
    return newList

def det_reward_func(custs, veh, task_list):

    late_cost = 4
    pending_cost = 2
    late_discount = 2

    task_list = task_list + [0]
    rewards, done = 0, False
    batch_size, nodes_count, d  = custs.size()
    custs = np.copy(custs[0,:,:])
    vehicles = np.zeros((5,))
    vehicles[:2] = custs[0,:2]
    vehicles[2:4] = np.copy(veh)

    for cust_idx in task_list:
        dest = custs[cust_idx,:]
        dist = np.linalg.norm(dest[:2]-vehicles[:2], axis=-1)
        tt = dist / vehicles[3]
        arv = np.max([vehicles[4] + tt, dest[4]])
        late = np.clip( arv - dest[5], 0, None)
        vehicles[:2] = dest[:2]
        vehicles[2] -= dest[2]
        vehicles[4] = arv + dest[6]

        # rewards += -dist - np.clip(late_cost*late, 0, pending_cost)
        # finish = (arv <= dest[5])
        # reward += dest[3]* finish - pending_cost* (1-finish) 
        finish = 1
        rewards += dest[3]* finish* np.exp(-late_discount*late)
        if cust_idx == 0:
            done = True
    return rewards

if __name__=='__main__':

    count_custs = 10
    count_vehs = count_custs//5
    DEFAULT_BID = 0

    data_path = "./data_sample10_stw/s_cvrptw_n{}m{}/norm_data.pyth".format(count_custs, count_vehs)    
    data = torch.load(data_path)
    loader = DataLoader(data, batch_size = 1)

    for batch in tqdm(loader):
        vehs, custs = batch
        CBBA_Params_N = np.size(vehs,1)
        CBBA_Params_M = np.size(custs,1) - 2
        CBBA_Data = {}
        CBBA_Data['path'] = -np.ones(CBBA_Params_M, dtype=int)
        CBBA_Data['scores'] = -np.ones(CBBA_Params_M)

        margin_gain_last_cust = []

        bundle_full = len(np.argwhere(CBBA_Data['path']==-1)) == 1 # when all nodes except depot are in bundle
        still_bidding = 1
        while (not bundle_full):
            L = np.argwhere(CBBA_Data['path']==-1)
            CBBA_Data['bids'] = DEFAULT_BID*np.ones(CBBA_Params_M)
            CBBA_Data['real_bids'] = DEFAULT_BID*np.ones(CBBA_Params_M)
            bestIdxs = np.zeros(CBBA_Params_M)
            taskTimes = np.zeros(CBBA_Params_M)
            for m in range(1, CBBA_Params_M):
                if not len(np.argwhere(CBBA_Data['path']==m)):
                    bestBid = DEFAULT_BID
                    bestIdx = 0
                    for j in range(L[0][0]+1):
                        skip = 0
                        taskPrev = [] if j==0 else CBBA_Data['path'][:j].tolist()
                        taskNext = [] if j==L[0][0] else CBBA_Data['path'][j:L[0][0]].tolist()
                        task_list = taskPrev + [m] +taskNext
                        reward = det_reward_func(custs, vehs[:,0,:], task_list)

                        original_score = np.sum(CBBA_Data['scores'][:L[0][0]])
                        score = reward - original_score
                        # print(f'{score}\t {score_old}')
                        if score>bestBid:
                            bestBid = score
                            bestIndex = j

                    if bestBid > DEFAULT_BID:
                        # Submodular Wrapper
                        existingBids = CBBA_Data['scores'][:L[0][0]]
                        bestBid_wrapped = min(np.append(existingBids,bestBid))
                        CBBA_Data['real_bids'][m] = bestBid
                        CBBA_Data['bids'][m] = bestBid_wrapped
                        bestIdxs[m] = bestIndex
            
            print('bids', CBBA_Data['bids'])
            print('Real_bids', CBBA_Data['real_bids'])
            value = max(CBBA_Data['bids'])
            bestTask = np.argwhere(CBBA_Data['bids']==value)[0]
            CBBA_Data['path'] = CBBA_InsertInList(CBBA_Data['path'], int(bestTask), bestIdxs[bestTask])
            CBBA_Data['scores'] = CBBA_InsertInList(CBBA_Data['scores'], CBBA_Data['real_bids'][bestTask], bestIdxs[bestTask])
            bundle_full = len(np.argwhere(CBBA_Data['path']==-1)) == 1       
            print('CBBA_scores:',CBBA_Data['scores'])
            # calcualte the Margin gain for the last customer
            m = CBBA_Params_M
            L = np.argwhere(CBBA_Data['path']==-1)
            bestBid = DEFAULT_BID
            bestIdx = 0
            for j in range(L[0][0]+1):
                skip = 0
                taskPrev = [] if j==0 else CBBA_Data['path'][:j].tolist()
                taskNext = [] if j==L[0][0] else CBBA_Data['path'][j:L[0][0]].tolist()
                task_list = taskPrev + [m] +taskNext
                reward = det_reward_func(custs, vehs[:,0,:], task_list)
                original_score = np.sum(CBBA_Data['scores'][:L[0][0]])
                score = reward - original_score
                # print(f'{score}\t {score_old}')
                if score>bestBid:
                    bestBid = score
                    bestIndex = j
            print('current path:{}  current score: {:10.4g}   margin gain of the last customer: {:10.4g}'\
                .format(CBBA_Data['path'], original_score, bestBid))
            
            margin_gain_last_cust.append(bestBid)
        fig, ax = plt.subplots()
        ax.plot(margin_gain_last_cust)       
        ax.set(title='marginal gain for the last customer')
        plt.show()

