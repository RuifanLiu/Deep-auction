# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:16:26 2020

@author: s313488
"""
import numpy as np
from ._score import Scoring_CalcScore_Original, Scoring_CalcScore_DNN, Scoring_CalcScore_Maxmin_DNN


class CBBA():
    
    def __init__(self, env, data, scorefun, **kwargs):
        # define CBBA constants
        self.env = env
        vehs, custs = data
        self.CBBA_Params_N = np.size(vehs,1) # number of agents
        self.CBBA_Params_M = np.size(custs,1) #number of tasks (exclude the depot)
        self.CBBA_Params_MAXDEPTH = np.size(custs,1) #maximum bundle depth (remember to set it for each scenario)
        
        self.CBBA_Data = []
        for n in range(self.CBBA_Params_N):
            info = {}
            info['agentID']    = n
            info['agentIndex'] = n
            info['bundle']     = -np.ones(self.CBBA_Params_MAXDEPTH, dtype=int)
            info['path']       = -np.ones(self.CBBA_Params_MAXDEPTH, dtype=int)
            info['times']      = -np.ones(self.CBBA_Params_MAXDEPTH, dtype=int)
            info['scores']     = -np.ones(self.CBBA_Params_MAXDEPTH)
            info['bids']       = np.zeros(self.CBBA_Params_M)
            info['winners']    = -np.ones(self.CBBA_Params_M)
            info['winnerBids'] = np.zeros(self.CBBA_Params_M)
            self.CBBA_Data.append(info)

        self.Scoring_CalcScore = scorefun
        self.vehs = vehs
        self.custs = custs
        self.value_model = kwargs.get('value_model') if 'value_model' in kwargs else None       
        self.value_model2 = kwargs.get('value_model2') if 'value_model2' in kwargs else None
        self.DRL_model = kwargs.get('DRL_model') if 'DRL_model' in kwargs else None
        
    def CBBA_Main(self):
        
        T = 1
        t = np.zeros([self.CBBA_Params_N, self.CBBA_Params_N])
        lastTime = T-1
        doneFlag = 0
        
        while(doneFlag==0):
            #---------------------------------------%
            # 1. Communicate
            # Perform consensus on winning agents and bid values (synchronous)
            self.CBBA_Communicate(t,T)
            # ---------------------------------------%
            #  2. Run CBBA bundle building/updating
            # ---------------------------------------%
            #  Run CBBA on each agent (decentralized but synchronous)            
            for n in range(self.CBBA_Params_N):
                newBid = self.CBBA_Bundle(n)
                if newBid:
                    lastTime = T
        
            # %---------------------------------------%
            # % 3. Convergence Check
            # %---------------------------------------%
            # % Determine if the assignment is over (implemented for now, but later
            # % this loop will just run forever)
            if(T-lastTime > self.CBBA_Params_N):
                doneFlag   = 1
            elif(T-lastTime > 2*self.CBBA_Params_N):
                print('Algorithm did not converge due to communication trouble');
                doneFlag = 1
            elif T>20:
                print('CBBA Loop exceed 20');
                self.CBBA_Communicate(t,T)
                for n in range(self.CBBA_Params_N):
                    self.CBBA_BundleRemove(n)
                doneFlag = 1
            else:
                #% Maintain loop
                T = T + 1
                # print(T)
        # Map path and bundle values to actual task indices
        # for n in range(self.CBBA_Params_N):
        #     for m in range(self.CBBA_Params_MAXDEPTH):
        #         if self.CBBA_Data[n]['bundle'][m] == -1:
        #             break
        #         else
        #             self.CBBA_Data[n]['bundle'][m] = tasks(CBBA_Data(n).bundle(m)).id;
        #         end
                
        #         if(CBBA_Data(n).path(m) == -1),
        #             break;
        #         else
        #             CBBA_Data(n).path(m) = tasks(CBBA_Data(n).path(m)).id;
        #         end
        #     end
        # end
        
        # Compute the total score of the CBBA assignment
        Total_Score = 0
        for n in range(self.CBBA_Params_N):
            # print(self.CBBA_Data[n]['scores'])
            for m in range(self.CBBA_Params_MAXDEPTH):
                if self.CBBA_Data[n]['bundle'][m]>-1:
                    Total_Score = Total_Score + self.CBBA_Data[n]['scores'][m]
                else:
                    break
        
        Assignment_result = [self.CBBA_Data[n]['path'] for n in range(self.CBBA_Params_N)]
        return Assignment_result, Total_Score       
    
    
        
    def CBBA_Communicate(self, old_t, T):
        # Copy data
        old_z = []
        old_y = []
        for n in range(self.CBBA_Params_N):
            old_z.append(self.CBBA_Data[n]['winners'])
            old_y.append(self.CBBA_Data[n]['winnerBids']) 
            
        old_z = np.array(old_z,dtype=int)
        old_y = np.array(old_y)
        
        z = old_z
        y = old_y
        t = old_t
        
        epsilon = 1e-40
        
        # Start communication between agents
        # % sender   = k
        # % receiver = i
        # % task     = j
        
        for k in range(self.CBBA_Params_N):
            for i in range(self.CBBA_Params_N):
                for j in range(self.CBBA_Params_M):
    
                    # Implement table for each task
                    if old_z[k,j] == k: # Entries 1 to 4: Sender thinks he has the task
                    
                        # Entry 1: Update or Leave
                        if z[i,j] == i: 
                            if old_y[k,j] - y[i,j] > epsilon:  # Update
                                z[i,j] = old_z[k,j]
                                y[i,j] = old_y[k,j]
                            elif abs(old_y[k,j] - y[i,j]) <= epsilon:  # Equal scores
                                if z[i,j] > old_z[k,j]:  # Tie-break based on smaller index
                                    z[i,j] = old_z[k,j]
                                    y[i,j] = old_y[k,j]
                            
                        # Entry 2: Update
                        elif z[i,j] == k : 
                            z[i,j] = old_z[k,j]
                            y[i,j] = old_y[k,j]
                        
                        # Entry 3: Update or Leave
                        elif z[i,j] > -1: 
                            if old_t[k,z[i,j]] > t[i,z[i,j]]:  # Update
                                z[i,j] = old_z[k,j]
                                y[i,j] = old_y[k,j]
                            elif old_y[k,j] - y[i,j] > epsilon:   # Update
                                z[i,j] = old_z[k,j]
                                y[i,j] = old_y[k,j]
                            elif abs(old_y[k,j] - y[i,j]) <= epsilon:  # Equal scores
                                if z[i,j] > old_z[k,j]:  # Tie-break based on smaller index
                                    z[i,j] = old_z[k,j]
                                    y[i,j] = old_y[k,j]

                        # Entry 4: Update
                        elif z[i,j] == -1:
                            z[i,j] = old_z[k,j]
                            y[i,j] = old_y[k,j]
                        else:
                            print['Unknown winner value: Should not be here, please revise']

                        
                    elif old_z[k,j] == i: # Entries 5 to 8: Sender thinks receiver has the task
    
                        # Entry 5: Leave
                        if z[i,j] == i: 
                            None  
                        # Entry 6: Reset
                        elif z[i,j] == k: 
                            z[i,j] = -1
                            y[i,j] = 0
                        # Entry 7: Reset or Leave
                        elif z[i,j] > -1: 
                            if old_t[k,z[i,j]] > t[i,z[i,j]]:  # Reset
                                z[i,j] = -1
                                y[i,j] = 0
                            
                        # Entry 8: Leave
                        elif z[i,j] == -1:
                            None
                        else:
                            print('Unknown winner value: Should not be here, please revise')
                        
                    elif old_z[k,j] > -1: # Entries 9 to 13: Sender thinks someone else has the task
                        
                        # Entry 9: Update or Leave
                        if z[i,j] == i :
                            if old_t[k,old_z[k,j]] > t[i,old_z[k,j]] :
                                if [old_y[k,j] - y[i,j]] > epsilon :
                                    z[i,j] = old_z[k,j]  # Update
                                    y[i,j] = old_y[k,j]
                                elif abs(old_y[k,j] - y[i,j]) <= epsilon: # Equal scores
                                    if z[i,j] > old_z[k,j]:  # Tie-break based on smaller index
                                        z[i,j] = old_z[k,j]
                                        y[i,j] = old_y[k,j]
                            
                        # Entry 10: Update or Reset
                        elif z[i,j] == k:
                            if old_t[k,old_z[k,j]] > t[i,old_z[k,j]]:  # Update
                                z[i,j] = old_z[k,j]
                                y[i,j] = old_y[k,j]
                            else:  # Reset
                                z[i,j] = -1
                                y[i,j] = 0
                            
                        # Entry 11: Update or Leave
                        elif z[i,j] == old_z[k,j]:
                            if old_t[k,old_z[k,j]] > t[i,old_z[k,j]]:  # Update
                                z[i,j] = old_z[k,j]
                                y[i,j] = old_y[k,j]
                            
                        # Entry 12: Update, Reset or Leave
                        elif z[i,j] > -1: 
                            if old_t[k,z[i,j]] > t[i,z[i,j]]:
                                if old_t[k,old_z[k,j]] >= t[i,old_z[k,j]]:  # Update
                                    z[i,j] = old_z[k,j]
                                    y[i,j] = old_y[k,j]
                                elif old_t[k,old_z[k,j]] < t[i,old_z[k,j]]: # Reset
                                    z[i,j] = -1
                                    y[i,j] = 0
                                else:
                                    print('Should not be here, please revise')

                            else:
                                if old_t[k,old_z[k,j]] > t[i,old_z[k,j]]:
                                    if [old_y[k,j] - y[i,j]] > epsilon:  # Update
                                        z[i,j] = old_z[k,j]
                                        y[i,j] = old_y[k,j]
                                    elif abs(old_y[k,j] - y[i,j]) <= epsilon:  # Equal scores
                                        if z[i,j] > old_z[k,j]:   # Tie-break based on smaller index
                                            z[i,j] = old_z[k,j]
                                            y[i,j] = old_y[k,j]
    
                        # Entry 13: Update or Leave
                        elif z[i,j] == -1:
                            if old_t[k,old_z[k,j]] > t[i,old_z[k,j]]:  # Update
                                z[i,j] = old_z[k,j]
                                y[i,j] = old_y[k,j]       
                        else:
                            print('Unknown winner value: Should not be here, please revise')
                        
                    elif old_z[k,j] == -1: # Entries 14 to 17: Sender thinks no one has the task
    
                        # Entry 14: Leave
                        if z[i,j] == i: 
                            None
                        # Entry 15: Update
                        elif z[i,j] == k:
                            z[i,j] = old_z[k,j]
                            y[i,j] = old_y[k,j]
                        
                        # Entry 16: Update or Leave
                        elif z[i,j] > -1:
                            if old_t[k,z[i,j]] > t[i,z[i,j]]:  # Update
                                z[i,j] = old_z[k,j]
                                y[i,j] = old_y[k,j]                                
                        # Entry 17: Leave
                        elif z[i,j] == -1:
                            None
                        else:
                            print('Unknown winner value: Should not be here, please revise')
                        # End of table
                        
                    else:
                        print('Unknown winner value: Should not be here, please revise')
                
                # Update timestamps for all agents based on latest comm
                for n in range(self.CBBA_Params_N):
                    if n != i and t[i,n] < old_t[k,n]:
                        t[i,n] = old_t[k,n]
                t[i,k] = T                            
        # Copy data
        for n in range(self.CBBA_Params_N):
            self.CBBA_Data[n]['winners']  = z[n,:]
            self.CBBA_Data[n]['winnerBids'] = y[n,:]
        
        
    def CBBA_Bundle(self, n):
        self.CBBA_BundleRemove(n)
        newBid = self.CBBA_BundleAdd(n)
        return newBid     
        
    def CBBA_BundleRemove(self,n):
        outbidForTask = 0;
        for j in range(self.CBBA_Params_MAXDEPTH):
            # If bundle(j) < 0, it means that all tasks up to task j are
            # still valid and in paths, the rest (j to MAX_DEPTH) are
            # released
            if self.CBBA_Data[n]['bundle'][j] < 0:
                # disp('Order is negative, breaking');
                break
            else:
                #  Test if agent has been outbid for a task.  If it has,
                #  release it and all subsequent tasks in its path.
                if self.CBBA_Data[n]['winners'][self.CBBA_Data[n]['bundle'][j]] != self.CBBA_Data[n]['agentIndex']:
                    outbidForTask = 1;        
                if outbidForTask:
                    #  The agent has lost a previous task, release this one too
                    if self.CBBA_Data[n]['winners'][self.CBBA_Data[n]['bundle'][j]] == self.CBBA_Data[n]['agentIndex']:
                        #  Remove from winner list if in there
                        self.CBBA_Data[n]['winners'][self.CBBA_Data[n]['bundle'][j]] = -1
                        self.CBBA_Data[n]['winnerBids'][self.CBBA_Data[n]['bundle'][j]] = 0
                    #  Clear from path and times vectors and remove from bundle
                    idx = np.argwhere(self.CBBA_Data[n]['path'] == self.CBBA_Data[n]['bundle'][j])
                    
                    self.CBBA_Data[n]['path']   = self.CBBA_RemoveFromList(self.CBBA_Data[n]['path'],   idx)
                    self.CBBA_Data[n]['times']  = self.CBBA_RemoveFromList(self.CBBA_Data[n]['times'],  idx)
                    self.CBBA_Data[n]['scores'] = self.CBBA_RemoveFromList(self.CBBA_Data[n]['scores'], idx)
        
                    self.CBBA_Data[n]['bundle'][j] = -1
        


        
    def CBBA_BundleAdd(self,n):
        epsilon = 1e-40
        newBid = 0
        CBBA_Data = self.CBBA_Data[n]
        bundleFull = len(np.argwhere(CBBA_Data['bundle']==-1)) == 1 # when all nodes except depot are in bundle
        value = 1
        while(value>0 and not bundleFull):
            CBBA_Data, bestIndxs, taskTimes, CBBA_double_bids = self.CBBA_ComputeBids(CBBA_Data,n)
            D1 = (CBBA_Data['bids'] - CBBA_Data['winnerBids'] > epsilon)
            D2 = (abs(CBBA_Data['bids'] - CBBA_Data['winnerBids']) <= epsilon)
            D3 = (CBBA_Data['agentIndex'] < CBBA_Data['winners'])

            if CBBA_double_bids is None:
                D = np.logical_or(D1, np.logical_and(D2, D3))
            else:
                D4 = (CBBA_double_bids - CBBA_Data['winnerBids'] > epsilon)
                D = np.logical_or(np.logical_and(D1, D4), np.logical_and(D2, D3))

            # D = D1
            value = max(np.multiply(D, CBBA_Data['bids']))
            bestTask = np.argwhere(np.multiply(D, CBBA_Data['bids'])==value)

            if value>0:
                newBid = 1
                allvalues = np.argwhere(np.multiply(D, CBBA_Data['bids'])==value)
                if len(allvalues) == 1:
                    bestTask = allvalues
                else:
                    bestTask = allvalues[0]
                

                if CBBA_double_bids is None:
                    CBBA_Data['winners'][bestTask] = CBBA_Data['agentIndex']
                    CBBA_Data['winnerBids'][bestTask] = CBBA_Data['bids'][bestTask]
                    
                    CBBA_Data['path'] = self.CBBA_InsertInList(CBBA_Data['path'], int(bestTask), bestIndxs[bestTask])
                    CBBA_Data['times'] = self.CBBA_InsertInList(CBBA_Data['times'], taskTimes[bestTask], bestIndxs[bestTask])
                    CBBA_Data['scores'] = self.CBBA_InsertInList(CBBA_Data['scores'], CBBA_Data['bids'][bestTask], bestIndxs[bestTask])
                    length = len(np.argwhere(CBBA_Data['bundle']>-1))
                    CBBA_Data['bundle'][length] = bestTask
                else:
                    # another value network for score vector
                    CBBA_Data['winners'][bestTask] = CBBA_Data['agentIndex']
                    CBBA_Data['winnerBids'][bestTask] = CBBA_double_bids[bestTask]
                    
                    CBBA_Data['path'] = self.CBBA_InsertInList(CBBA_Data['path'], int(bestTask), bestIndxs[bestTask])
                    CBBA_Data['times'] = self.CBBA_InsertInList(CBBA_Data['times'], taskTimes[bestTask], bestIndxs[bestTask])
                    CBBA_Data['scores'] = self.CBBA_InsertInList(CBBA_Data['scores'], CBBA_double_bids[bestTask], bestIndxs[bestTask])
                    length = len(np.argwhere(CBBA_Data['bundle']>-1))
                    CBBA_Data['bundle'][length] = bestTask

                
            bundleFull = len(np.argwhere(CBBA_Data['bundle']==-1)) == 1 
            # print('bundle:',CBBA_Data['bundle'])
            # print('CBBA_bid:',CBBA_Data['bids'])
            # print('CBBA_double_bid:', CBBA_double_bids)
            # print('CBBA_scores:', CBBA_Data['scores'])
            # print('CBBA_winnerbid:',CBBA_Data['winnerBids'])
        self.CBBA_Data[n] = CBBA_Data
        return newBid
        
        
    def CBBA_ComputeBids(self, CBBA_Data,n):
        L = np.argwhere(CBBA_Data['path']==-1)
        if len(L)==0:
            return
        CBBA_Data['bids'] = np.zeros(self.CBBA_Params_M)
        bestIdxs = np.zeros(self.CBBA_Params_M)
        taskTimes = np.zeros(self.CBBA_Params_M)

        CBBA_double_bids = None
        

        if self.Scoring_CalcScore == 'Scoring_CalcScore_Original' or self.Scoring_CalcScore == 'Scoring_CalcScore_Robust':
            if self.Scoring_CalcScore == 'Scoring_CalcScore_Original' :
                robust_greedy = False
            else:
                robust_greedy = True
            for m in range(1, self.CBBA_Params_M):
                if not len(np.argwhere(CBBA_Data['path']==m)):
                    bestBid = 0
                    bestIdx = 0
                    bestTime = -1
                    for j in range(L[0][0]+1):
                        skip = 0
                        if j==0:
                            taskPrev = []
                            timePrev = []
                        else:
                            taskPrev = CBBA_Data['path'][:j].tolist()
                            timePrev = CBBA_Data['times'][:j].tolist()
                        if j==L[0][0]:
                            taskNext = []
                            timeNext = []
                        else:
                            taskNext = CBBA_Data['path'][j:L[0][0]].tolist()
                            timeNext = CBBA_Data['times'][j:L[0][0]].tolist()
                        
                        score_old, score_new, minStart, maxStart = Scoring_CalcScore_Original(self.env, n, self.custs, self.vehs, m, 
                                                                                 taskPrev,timePrev,
                                                                                 taskNext,timeNext,
                                                                                 robust_greedy = robust_greedy)
                        original_score = np.sum(CBBA_Data['scores'][:L[0][0]])
                        score = score_new - original_score
                        # print(f'{score}\t {score_old}')

                        if minStart > maxStart:
                            skip=1
                        if not skip:
                            if score>bestBid:
                                bestBid = score
                                bestIndex = j
                                bestTime = minStart

                    if bestBid > 0:
                        # Submodular Wrapper
                        existingBids = CBBA_Data['scores'][:L[0][0]]
                        bestBid = min(np.append(existingBids,bestBid))

                        CBBA_Data['bids'][m] = bestBid
                        bestIdxs[m] = bestIndex
                        taskTimes[m] = bestTime
                
        elif self.Scoring_CalcScore == 'Scoring_CalcScore_DNN':

            unAllocatedTask = []
            for m in range(1, self.CBBA_Params_M):
                if not len(np.argwhere(CBBA_Data['path']==m)):
                    unAllocatedTask.append(m)
            j=L[0][0]

            taskPrev = []
            timePrev = []
            taskNext = CBBA_Data['path'][:j].tolist()
            timeNext = CBBA_Data['times'][:j].tolist()
                    
            score_old, score_new, minStart, maxStart = Scoring_CalcScore_DNN(self.env, n, self.custs, self.vehs, \
                self.value_model, unAllocatedTask, taskPrev, timePrev, taskNext,timeNext)
            original_score = np.sum(CBBA_Data['scores'][:L[0][0]])
            score = score_new - original_score

            # Submodular Wrapper
            existingBids = CBBA_Data['scores'][:L[0][0]]
            minExistingBids = min(existingBids) if len(existingBids) else 100
            score = [min([i,minExistingBids]) for i in score]
            
            idx = 0
            margin_thershold = -100
            for m in unAllocatedTask:
                if score[idx]>margin_thershold:
                    CBBA_Data['bids'][m] = score[idx]
                    bestIdxs[m] = 0
                    taskTimes[m] = 0
                idx = idx+1

        elif self.Scoring_CalcScore == 'Scoring_CalcScore_DNN_with_Insertion_Order':   
            
            unAllocatedTask = []
            for m in range(1, self.CBBA_Params_M):
                if not len(np.argwhere(CBBA_Data['path']==m)):
                    unAllocatedTask.append(m)
            j=L[0][0]

            taskPrev = []
            timePrev = []
            taskNext = CBBA_Data['path'][:j].tolist()
            timeNext = CBBA_Data['times'][:j].tolist()
                    
            score, minStart, maxStart = Scoring_CalcScore_DNN(self.env, n, self.custs, self.vehs, self.value_model, unAllocatedTask, taskPrev, timePrev, taskNext, timeNext)
            
            # Submodular Wrapper
            existingBids = CBBA_Data['scores'][:L[0][0]]
            minExistingBids = min(existingBids) if len(existingBids) else 100
            score = [min([i,minExistingBids]) for i in score]
            
            idx = 0
            margin_thershold = -100
            for m in unAllocatedTask:
                if score[idx]>margin_thershold:
                    CBBA_Data['bids'][m] = score[idx]
                    # bestIdxs[m] = 0
                    taskTimes[m] = 0
                idx = idx+1

            ###############################################   
            # find the best insertion place
            for m in range(1, self.CBBA_Params_M):
                if not len(np.argwhere(CBBA_Data['path']==m)):
                    bestBid = 0
                    bestIdx = L[0][0]
                    bestTime = -1
                    for j in range(L[0][0]+1):
                        skip = 0
                        if j==0:
                            taskPrev = []
                            timePrev = []
                        else:
                            taskPrev = CBBA_Data['path'][:j].tolist()
                            timePrev = CBBA_Data['times'][:j].tolist()
                        if j==L[0][0]:
                            taskNext = []
                            timeNext = []
                        else:
                            taskNext = CBBA_Data['path'][j:L[0][0]].tolist()
                            timeNext = CBBA_Data['times'][j:L[0][0]].tolist()
                        
                        score_old, score_new, minStart, maxStart = Scoring_CalcScore_Original(self.env, n, self.custs, self.vehs, m, 
                                                                                 taskPrev,timePrev,
                                                                                 taskNext,timeNext,
                                                                                 robust_greedy = False)
                        # L = np.argwhere(CBBA_Data['path']==-1)
                        # original_score = np.sum(CBBA_Data['scores'][:L[0][0]])

                        score = score_new - score_old
                        # print(f'{score}\t {score_old}')

                        if minStart > maxStart:
                            skip=1
                        if not skip:
                            if score>bestBid:
                                bestBid = score
                                bestIdx = j
                                bestTime = minStart
                    # if score > 0: task is inserted to the path in the best index
                    # if score <=0: task is appended to the path
                    bestIdxs[m] = bestIdx

        elif self.Scoring_CalcScore == 'Scoring_CalcScore_Maxmin_DNN':

            unAllocatedTask = []
            for m in range(1, self.CBBA_Params_M):
                if not len(np.argwhere(CBBA_Data['path']==m)):
                    unAllocatedTask.append(m)
            j=L[0][0]

            taskPrev = []
            timePrev = []
            taskNext = CBBA_Data['path'][:j].tolist()
            timeNext = CBBA_Data['times'][:j].tolist()
                    
            score_old, score_new,  minStart, maxStart = Scoring_CalcScore_Maxmin_DNN(n, self.custs, self.vehs, self.value_model, self.value_model2, unAllocatedTask, taskPrev, timePrev, taskNext,timeNext)
            original_score = np.sum(CBBA_Data['scores'][:L[0][0]])
            score = score_new - original_score
            
            # Submodular Wrapper
            existingBids = CBBA_Data['scores'][:L[0][0]]
            minExistingBids = min(existingBids) if len(existingBids) else 100
            score = [min([i,minExistingBids]) for i in score]
            
            idx = 0
            margin_thershold = -100
            for m in unAllocatedTask:
                if score[idx]>margin_thershold:
                    CBBA_Data['bids'][m] = score[idx]
                    bestIdxs[m] = 0
                    taskTimes[m] = 0
                idx = idx+1
        
        elif self.Scoring_CalcScore == 'Scoring_CalcScore_Double_DNN':
            CBBA_double_bids = np.zeros(self.CBBA_Params_M)
            unAllocatedTask = []
            for m in range(1, self.CBBA_Params_M):
                if not len(np.argwhere(CBBA_Data['path']==m)):
                    unAllocatedTask.append(m)
            j=L[0][0]

            taskPrev = []
            timePrev = []
            taskNext = CBBA_Data['path'][:j].tolist()
            timeNext = CBBA_Data['times'][:j].tolist()
                    
            score_old, score_new,  minStart, maxStart = Scoring_CalcScore_DNN(self.custs, self.vehs[n], self.value_model, unAllocatedTask, taskPrev, timePrev, taskNext,timeNext)
            original_score = np.sum(CBBA_Data['scores'][:L[0][0]])
            score = score_new - original_score
            
            score_old, score_new,  minStart, maxStart = Scoring_CalcScore_DNN(self.custs, self.vehs[n], self.value_model2, unAllocatedTask, taskPrev, timePrev, taskNext,timeNext)
            original_score = np.sum(CBBA_Data['scores'][:L[0][0]])
            score2 = score_new - original_score
            # Submodular Wrapper
            existingBids = CBBA_Data['scores'][:L[0][0]]
            minExistingBids = min(existingBids) if len(existingBids) else 100
            score = [min([i,minExistingBids]) for i in score]
            score2 = [min([i,minExistingBids]) for i in score2]

            idx = 0
            margin_thershold = -100
            for m in unAllocatedTask:
                if score[idx]>margin_thershold:
                    CBBA_Data['bids'][m] = score[idx]
                    CBBA_double_bids[m] = score2[idx]
                    bestIdxs[m] = 0
                    taskTimes[m] = 0
                idx = idx+1
        else:
            raise ValueError('No valid Reward Function')
        
        return CBBA_Data, bestIdxs, taskTimes, CBBA_double_bids
                    
        
        
    def CBBA_InsertInList(self, oldList, value, index):
        index = int(index)
        newList = np.copy(oldList)
        newList[:index] = oldList[:index]
        newList[index]       = value
        newList[index+1:] = oldList[index:-1]
        return newList
        
    def CBBA_RemoveFromList(self, oldList, index):
        index = int(index)
        newList = np.copy(oldList)
        newList[:index]  = oldList[:index]
        newList[index:-1] = oldList[index+1:]
        newList[-1] = -1
        return newList    
        
        
    