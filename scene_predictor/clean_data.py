from pathlib import Path
from glob import glob

import numpy as np
from tqdm import tqdm

import pickle

if __name__ == '__main__':
    # dest_path = Path(f'/common/users/dm1487/legged_manipulation_data/rollout_data/multi_policy_single_trajectories_1')
    # ctr = 0
    # data_ctr = {
    #     0: [],
    #     1: [],
    #     2: [],
    #     3: [],
    # }
    # for file in tqdm(sorted(glob(str(dest_path/'*/*.npz')))):
    #     data = np.load(file)

    #     try:
        
    #         last_idx = data['done'].nonzero()[0][-1]
    #         # print(data['fsw'][last_idx, 2], data['fsw'][last_idx, 9], data['fsw'][last_idx, 16])

    #         data_ctr[int(int(data['target'][last_idx, 2+6] != 0) + int(data['target'][last_idx, 9+6] != 0) + int(data['target'][last_idx, 16+6]!= 0))].append(file)
    #         # print(data['target'][last_idx, 2+6], data['target'][last_idx, 9+6], data['target'][last_idx, 16+6])
    #         ctr += 1
    #     except:
    #         print(file)
    #     # if ctr == 10000:
    #     #     break
    
    # print(len(data_ctr[0]), len(data_ctr[1]), len(data_ctr[2]), len(data_ctr[3]))
    # with open('./data_ctr_1.pkl', 'wb') as f:
    #     pickle.dump(data_ctr, f)

    with open('./data_ctr_1.pkl', 'rb') as f:
        data_ctr = pickle.load(f)

    balanced_data = []
    balanced_data += data_ctr[0]
    for i in range(1, 3):
        balanced_data += data_ctr[i][:]
    
    print(len(balanced_data))
    with open('./scene_predictor/balanced_data_1.pkl', 'wb') as f:
        pickle.dump(balanced_data, f)
        
