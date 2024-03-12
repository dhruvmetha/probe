from pathlib import Path
from glob import glob

import numpy as np
from tqdm import tqdm

import pickle
import os
import multiprocessing as mp

def remove_files(files, here):
    for file in tqdm(files):
        try:
            data = np.load(file)

        except:
            print('file_error', file)
            try:
                os.remove(file)
            except:
                pass
            continue
        
        try:
            if data['target'][10, 0] == 0. and data['target'][10, 1] == 0.:
                print('pose error', file)
                os.remove(file)

            data['done']
            data['target_env']
            data['target']
        except:
            print('pose error', file)
            os.remove(file)

def main(all_files):
    num_workers = 24
    # all_files = glob.glob(str(data_path/'*/*.npz'))
    print(len(all_files))

    files_per_worker = max(1, int(len(all_files)/num_workers))
    
    workers = []
    # each worker gets a subset of all_files
    for i in range(num_workers):

        start_idx = i*files_per_worker
        end_idx = (i+1)*files_per_worker
        # if i == num_workers-1:
        #     end_idx = len(all_files)
        files = all_files[start_idx:end_idx]

        workers.append(mp.Process(target=remove_files, args=(files, None)))
                     
    # workers = []
    # for folders in tqdm(sorted(glob.glob(str(data_path/'*')))):
    #     final_dest = Path(dest_path)/str((Path(folders).stem))
    #     final_dest.mkdir(parents=True, exist_ok=True)
    #     workers.append(mp.Process(target=build_traj, args=(folders, final_dest)))

    for worker in workers:
        worker.daemon = True
        worker.start()
    
    for worker in workers:
        worker.join()

if __name__ == '__main__':
    # dest_path = Path(f'/common/users/dm1487/legged_manipulation/rollout_data/exploration_single_trajectories')
    # files = sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data/exploration_3_single_trajectories1/*/*.npz')) + sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data/exploration_4_single_trajectories1/*/*.npz')) + sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data/exploration_6_single_trajectories1/*/*.npz'))

    # val_files = glob('/common/users/dm1487/legged_manipulation/rollout_data/no_exploration_1_single_trajectories/*/*.npz')

    # files = val_files
    # dest_path = Path(f'/common/users/dm1487/legged_manipulation/rollout_data/exploration_6_single_trajectories1')
    # dest_path = Path(f'/common/users/dm1487/legged_manipulation/rollout_data/exploration_3_single_trajectories1')
    # dest_path = Path(f'/common/users/dm1487/legged_manipulation/rollout_data/exploration_4_single_trajectories1')

    # files = glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/aug27/2_obs/1/*/*.npz')
    root_path = '/common/users/dm1487/legged_manipulation_data_store'
    root_traj_path = f'{root_path}/trajectories'
    sub_path = 'iros24_play_feb23_new_policy/2_obs'

    files = glob(f'{root_traj_path}/{sub_path}/*/*/*.npz')
    # with open('/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24/balanced/train_1.pkl', 'rb') as f:
    #     files = pickle.load(f)
    print(len(files))
    main(files)
    
    
    # exit()
    # ctr = 0
    # data_ctr = {
    #     0: [],
    #     1: [],
    #     2: [],
    #     3: [],
    # }
    # for file in tqdm(files):
    #     try:
    #         data = np.load(file)
    #         # last_idx = data['done'].nonzero()[0][-1]
    #         # data_ctr[int(int(data['target'][last_idx, 2+6] != 0) + int(data['target'][last_idx, 9+6] != 0) + int(data['target'][last_idx, 16+6]!= 0))].append(file)
    #         # ctr += 1
    #     except:
    #         print(file)
    #         os.remove(file)

    # files = glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/2_obs/2/*/*.npz')
    # print(len(files))
    
    # print(len(data_ctr[0]), len(data_ctr[1]), len(data_ctr[2]), len(data_ctr[3]))
    # with open('./scene_predictor/2_obs_data_store_files_clean_1.pkl', 'wb') as f:
    #     pickle.dump(data_ctr, f)

    # with open('./scene_predictor/data_ctr_7_priv_info.pkl', 'rb') as f:
    #     data_ctr = pickle.load(f)

    # print(data_ctr[0][0])
    # print(data_ctr[1][0])
    # print(data_ctr[2][0])
    

    # balanced_data = []
    # # balanced_data += data_ctr[0]
    # for i in range(0, 3):
    #     balanced_data += data_ctr[i][:17492]
    
    # print(len(balanced_data))
    # with open('./scene_predictor/balanced_data_7_priv_info.pkl', 'wb') as f:
    #     pickle.dump(balanced_data, f)
        
