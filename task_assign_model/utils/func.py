import numpy as np


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




def reward_func(obs, task_list, method):
    
    task_rewards = np.copy(obs[0,:]) # 0 reward 1 task_duration 2 start_time 3 visit_flag
    task_durations = np.copy(obs[1,:])
    task_startTime = np.copy(obs[2,:])
    task_visitflag = np.copy(obs[3,:]) # 0 pending 1 visited
    system_time = np.copy(obs[4,:])[0]

    time_discount = 0.9
    UNCERTAINTY_sigma = 1.0

    if method == 'greedy':

        rewards = 0
        for ii in range(len(task_list)):
            task_no = int(task_list[ii])
            
            task_duration_mean = task_durations[task_no]
            task_duration = task_duration_mean

            task_executionTime = np.max([task_startTime[task_no], system_time])
            task_reward = task_rewards[task_no]*np.exp(-time_discount*(task_executionTime-task_startTime[task_no]))
            reward = task_reward if not task_visitflag[task_no] else 0.0    

            system_time = task_executionTime + task_duration
            task_visitflag[task_no] = 1 # task is masked once visited

            rewards = rewards + reward
            
        output_reward = rewards
        output_time = system_time

    elif method == 'robust greedy':
        
        n_sample = 100
        #rewardss,times = [],[]
        #for _ in range(n_sample):
        rewards = 0
        system_time = [system_time]
        for ii in range(len(task_list)):
            task_no = int(task_list[ii])
            task_duration_mean = task_durations[task_no]
            task_duration = np.random.normal(task_duration_mean, UNCERTAINTY_sigma, n_sample)
            # task_duration = np.random.normal(task_duration_mean, UNCERTAINTY_sigma)
            task_duration = np.clip(task_duration, 0, task_duration_mean*2)

            try:
                task_executionTime = np.array([np.max([task_startTime[task_no], sys_time]) for sys_time in system_time])
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")

            task_reward = task_rewards[task_no]*np.exp(-time_discount*(task_executionTime-task_startTime[task_no]))
            reward = task_reward if not task_visitflag[task_no] else 0.0    

            system_time = task_executionTime + task_duration
            task_visitflag[task_no] = 1 # task is masked once visited

            rewards = rewards + reward

        output_reward = np.mean(rewards)
        output_time = np.mean(system_time)
            
    
    return output_reward
