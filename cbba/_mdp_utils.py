from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm
from multiprocessing import Pool
# constructs mdps and solve it
import time


class MDP():
    def __init__(self, batch, horizon=480, speed = 1.0, speed_var = 0.2, pending_cost =1, loc_scl=100, t_scl=480) -> None:
        self.vehs = np.array(batch[0][0])
        self.nodes = np.array(batch[1][0])
        self.vehs_count = np.size(self.vehs, 0)
        self.nodes_count = np.size(self.nodes, 0)

        self.horizon = horizon 
        self.speed = speed
        self.speed_var = speed_var
        self.pending_cost = pending_cost
        self.loc_scl = loc_scl
        self.t_scl = t_scl

    def mdp_generation(self):

        output_TP = []
        output_reward = []

        end_state = {
            'time': self.horizon,
            'available_task': np.ones(self.nodes_count),
            'cur_node': self.nodes_count-1,
        }
        end_state['available_task'][0] = 0 


# def ort_solve(data, late_cost = 1):
#     with Pool() as p:
#         with tqdm(desc = "Calling ORTools", total = data.batch_size) as pbar:
#             results = [p.apply_async(_solve_cp, (nodes, data.vehs, late_cost),
#                 callback = lambda _:pbar.update()) for nodes in data.nodes_gen()]
#             routes = [res.get() for res in results]
#     return routes

        num_total_state = self.state_to_idx(end_state)
        for iact in range(self.nodes_count):
            row_idxes, col_idxes, tp_values, rew_values = [],[],[],[]
            # t_start = time.perf_counter()
            with Pool() as p:
                with tqdm(desc = "Cpnstructing MDPs for act {}".format(iact), total = num_total_state) as pbar:
                    async_results = [p.apply_async(self._mdp_transition, (row_idx, iact),
                     callback = lambda _:pbar.update()) for row_idx in range(num_total_state)]
                    results = [ar.get() for ar in async_results]
            # print('time_pool:'+str(time.perf_counter()-t_start))
            results = np.concatenate(results)
            row_idxes = results[:,0]
            col_idxes = results[:,1]
            tp_values = results[:,2]
            rew_values = results[:,3]
        #     for res in results:
            
        # # for iact in range(self.nodes_count):
        # #     row_idxes, col_idxes, tp_values, rew_values = [],[],[],[]
        #     # for row_idx in tqdm(range(num_total_state), desc = "Cpnstructing MDPs for act = {}".format()):
        #     #     res = self._mdp_transition(row_idx, iact)
        #         row_idx, col_idx, poss, rew = res
        #         row_idxes = row_idxes + row_idx
        #         col_idxes = col_idxes + col_idx
        #         tp_values = tp_values + poss
        #         rew_values = rew_values + rew
            # print('get result:'+str(time.perf_counter()-t_start))
            TP = csr_matrix((tp_values,(row_idxes, col_idxes)), shape=(num_total_state, num_total_state), dtype=np.float64)
            Reward = csr_matrix((rew_values,(row_idxes, col_idxes)), shape=(num_total_state, num_total_state), dtype=np.float64)
            # print('generate matrix:'+str(time.perf_counter()-t_start))
            output_TP.append(TP)
            output_reward.append(Reward)
        return output_TP, output_reward

    def _mdp_transition(self, row_idx, iact):
        output = []
        is_state, state = self.idx_to_state(row_idx)
        if not is_state: return []
        if row_idx == 2886 and iact == 1: 
            print('1')
        avail_tasks, times, rews, posses = self.state_transition(state, iact)
        for ipos in range(len(posses)):
            next_state = {
                'time': times[ipos],
                'available_task': avail_tasks,
                'cur_node': iact
            }
            col_idx = self.state_to_idx(next_state)
            output.append([row_idx, col_idx, posses[ipos], rews[ipos]])
        return np.array(output)

    def state_to_idx(self, state):
    # index encoding
    # index = ((task_binary)*env.horizon + time)*(Nt+1) + cur_node
    # task_binary = [0 1 0 1 ...] = [0, 2^Nt], 0 unavailable task; 1 available task
        dec_avail_task = 0
        node_num = len(state['available_task'])

        if sum(state['available_task']):
            for task_no in range(node_num):
                dec_avail_task = dec_avail_task + state['available_task'][task_no]*2**(node_num-1-task_no)

        index = (dec_avail_task*(self.horizon+1) + state['time'])*(self.nodes_count) + state['cur_node']
        return int(index)

    def idx_to_state(self, idx):

        cur_node = idx % (self.nodes_count)
        time = int(idx / (self.nodes_count)) % (self.horizon+1)
        dec_avail_task = int(int(idx / (self.nodes_count)) / (self.horizon+1))

        available_task = np.zeros(self.nodes_count)
        for task_no in range(self.nodes_count):
            is_task_in = int(dec_avail_task / 2**(self.nodes_count-1-task_no))
            if is_task_in:
                available_task[task_no] = 1
                dec_avail_task = int(dec_avail_task - 2**task_no)
        if time > self.horizon or cur_node >= self.nodes_count:
            is_state = False
            state = []
        else:
            is_state = True
            state = {
                    'time': time,
                    'available_task': available_task,
                    'cur_node': cur_node,
                }
        return is_state, state

    def state_transition(self, state, iact):
        avail_tasks = np.copy(state['available_task'])

        if (avail_tasks[iact] == 1) and (iact is not 0) and (iact is not state['cur_node']):
            avail_tasks[iact] = 0
            dist = np.linalg.norm(self.nodes[state['cur_node']][:2]-self.nodes[iact][:2], axis=-1)
            dist = dist*self.loc_scl

            max_speed = self.speed*1.5
            min_speed = self.speed*0.5
            max_time_path = int(dist/min_speed)
            min_time_path = int(dist/max_speed)

            internal_times_1 = range(min_time_path, max_time_path)
            internal_times_2 = range(min_time_path+1, max_time_path+1)
            speed_nm = norm(loc = self.speed, scale = self.speed_var)
            posses = speed_nm.cdf(dist/internal_times_1) - speed_nm.cdf(dist/internal_times_2)
            posses = posses/sum(posses)

            tt = np.array(internal_times_2)
            time_arrival = state['time'] + tt
            time_start = np.array([max([arv, self.nodes[iact][4]]*self.t_scl) for arv in time_arrival]) 
            late = np.array([arv > self.nodes[iact][5]*self.t_scl for arv in time_arrival])
            rews = self.nodes[iact][3] * (1-late) - self.pending_cost*late
            times = time_start + self.nodes[iact][6]*self.t_scl
            exceed_horizon = np.where(times > self.horizon)
            rews[exceed_horizon] = 0.0
            times[exceed_horizon] = self.horizon
                
        elif iact is 0:
            avail_tasks[iact] = 0
            times = [self.horizon]
            rews = [0.0]
            posses = [1.0]
        else:
            avail_tasks[iact] = 0
            times = [state['time'] + self.nodes[iact][6]*self.t_scl]
            rews = [-1.0]
            posses = [1.0]

        return avail_tasks, times, rews, posses

