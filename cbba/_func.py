import os
import numpy as np
from itertools import  zip_longest

def export_cbba_rewards(output_dir, header, cbba_stats, dnn_cbba_stats):
    fpath = os.path.join(output_dir, "cbba_result.csv")
    with open(fpath, 'a') as f:
        f.write( (' '.join("{: >16}" for _ in range(9)) + '\n').format(
            "#(AGT,TASK)", "#DNN_EXP", "#DNN_ACT", "#CBBA_EXP", "#CBBA_ACT", "#CBBA_ACT_DRL"
            ))
        for (hdr, cbba, dnn_cbba) in enumerate(zip_longest(header, cbba_stats, dnn_cbba_stats, fillvalue=float('nan')), start = start_ep):
                f.write( ("{: >16d}" + ' '.join("{: >16.3g}" for _ in range(5)) + '\n').format(
                    hdr, *dnn_cbba, *cbba))

def path_construction(obs, method):
    # observation shape (n_env, feat_size, node_size)
    n_env, feat_size, node_size = obs.shape
    assert n_env == 1, "error!, doesn't support paralel environments"
    obs = obs.reshape((feat_size, node_size))

    task_path = []
    pending_tasks = np.array(range(node_size))
    bestTask = 0
    bestScore = 0
    bestPath = []

    for _ in range(node_size):

        for task in pending_tasks:
            length = len(task_path)
            for j in range(length+1):
                taskPrev = [] if j==0 else task_path[:j]
                taskNext = [] if j==len(task_path) else task_path[j:]

                reward = reward_func(obs, taskPrev+[task]+taskNext, method)
                
                if reward > bestScore:
                    bestTask = task
                    bestScore = reward
                    bestPath = taskPrev+[task]+taskNext
                    

        task_path = bestPath
        pending_tasks = np.delete(pending_tasks, np.where(pending_tasks == bestTask))
        # print(f'{task_path}')
        # print(f'{pending_tasks}')
        # print(f'{bestTask}')
        # print(f'{bestScore}')

    return task_path, bestScore
