from pathlib import Path
from glob import glob

import numpy as np
from tqdm import tqdm

import pickle

if __name__ == '__main__':
    # dest_path = Path(f'/common/users/dm1487/legged_manipulation/rollout_data/exploration_single_trajectories')
    # files = sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data/exploration_3_single_trajectories1/*/*.npz')) + sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data/exploration_4_single_trajectories1/*/*.npz')) + sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data/exploration_6_single_trajectories1/*/*.npz'))

    # val_files = glob('/common/users/dm1487/legged_manipulation/rollout_data/no_exploration_1_single_trajectories/*/*.npz')

    # files = val_files
    # # dest_path = Path(f'/common/users/dm1487/legged_manipulation/rollout_data/exploration_6_single_trajectories1')
    # # dest_path = Path(f'/common/users/dm1487/legged_manipulation/rollout_data/exploration_3_single_trajectories1')
    # # dest_path = Path(f'/common/users/dm1487/legged_manipulation/rollout_data/exploration_4_single_trajectories1')
    # ctr = 0
    # data_ctr = {
    #     0: [],
    #     1: [],
    #     2: [],
    #     3: [],
    # }
    # for file in tqdm(files):
    #     data = np.load(file)
    #     try:
    #         last_idx = data['done'].nonzero()[0][-1]
    #         data_ctr[int(int(data['target'][last_idx, 2+6] != 0) + int(data['target'][last_idx, 9+6] != 0) + int(data['target'][last_idx, 16+6]!= 0))].append(file)
    #         ctr += 1
    #     except:
    #         print(file)
    
    # print(len(data_ctr[0]), len(data_ctr[1]), len(data_ctr[2]), len(data_ctr[3]))
    # with open('./scene_predictor/data_ctr_5_val.pkl', 'wb') as f:
    #     pickle.dump(data_ctr, f)

    with open('./scene_predictor/data_ctr_5_val.pkl', 'rb') as f:
        data_ctr = pickle.load(f)
    

    balanced_data = []
    # balanced_data += data_ctr[0]
    for i in range(0, 3):
        balanced_data += data_ctr[i][:1000]
    
    print(len(balanced_data))
    with open('./scene_predictor/balanced_data_5_val.pkl', 'wb') as f:
        pickle.dump(balanced_data, f)
        
