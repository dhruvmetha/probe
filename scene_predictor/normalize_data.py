from pathlib import Path
from glob import glob

import numpy as np
from tqdm import tqdm

import pickle
import random
import os
import multiprocessing as mp
import shutil
        

def balance_data(files, dest_path):
    ctr = 0
    data_ctr = {
        0: [],
        1: [],
        2: [], 
        'right': [],
        'left': [],
    }

    data_2 = {
        'left': {
            'right': [],
            'left': [],
        },
        'right': {
            'right': [],
            'left': [],
        }
    }

    data_1 = {
        'mov': {
            'right': [],
            'left': [],
        },
        'imm': {
            'right': [],
            'left': [],
        }
    }

    non_files = []
    mv_files = []
    imm_files = []
    
    for file in tqdm(files):
        try:
            data = np.load(file)
            last_idx = data['done'].nonzero()[0][-1]
            target = data['target_env']
            index_found = int(int(target[last_idx, 2] != 0) + int(target[last_idx, 9] != 0)) # + int(target[last_idx, 16]!= 0))
            data_ctr[index_found].append(file)

            if index_found == 1:
                if int(target[last_idx, 2] != 0):
                    mv_files.append(file)
                    if target[last_idx, 3] > 0:
                        # right side immovable
                        data_1['mov']['left'].append(file)
                    else:
                        # left side immovable
                        data_1['mov']['right'].append(file)
                else:
                    imm_files.append(file)
                    if target[last_idx, 10] > 0:
                        # right side immovable
                        data_1['imm']['left'].append(file)
                    else:
                        # left side immovable
                        data_1['imm']['right'].append(file)

            # if index_found == 2:
            #     if target[last_idx, 3] > 0:
            #         if target[last_idx, 10] > 0:
            #             # right side immovable
            #             data_2['left']['left'].append(file)
            #         else:
            #             # left side immovable
            #             data_2['left']['right'].append(file)
            #     else:
            #         if target[last_idx, 10] > 0:
            #             # right side immovable
            #             data_2['right']['left'].append(file)
            #         else:
            #             # left side immovable
            #             data_2['right']['right'].append(file)

            ctr += 1
        except Exception as e:
            print(e)
            # print(file)
    
    # full_data = {
    #     # 'mv': mv_files,
    #     # 'imm': imm_files,
    #     # 'non': non_files,
    #     'data_ctr': data_ctr,
    # }

    

    # print(len(mv_files), len(imm_files), len(non_files))
   
    # print(len(data_ctr[0]), len(data_ctr[1]))
    
    # count = min(len(data_1['mov']['left']), len(data_1['mov']['right']))

    # print(len(data_ctr[0]))

    # print(len(data_1['mov']['left']), len(data_1['mov']['right']), len(data_1['imm']['left']), len(data_1['imm']['right']))



    # count = min(len(data_1['imm']['left']), len(data_1['imm']['right']), len(data_1['mov']['left']), len(data_1['mov']['right'])) # , len(data_2['right']['left']), len(data_2['right']['right']), len(data_2['left']['left']), len(data_2['left']['right']))
    
    # balance_data_mov_1 = random.sample(data_1['mov']['left'], min(count*2, len(data_1['mov']['left']))) + random.sample(data_1['mov']['right'], min(count*2, len(data_1['mov']['right'])))

    # balance_data_imm_1 = random.sample(data_1['imm']['left'], min(count*2, len(data_1['imm']['left']))) + random.sample(data_1['imm']['right'], min(count*2, len(data_1['imm']['right'])))
    
    # balance_data_2 = random.sample(data_2['left']['left'], min(count, len(data_2['left']['left']))) + random.sample(data_2['left']['right'], min(count, len(data_2['left']['right']))) + random.sample(data_2['right']['left'], min(count, len(data_2['right']['left']))) + random.sample(data_2['right']['right'], min(count, len(data_2['right']['right'])))
    
    
    # data = {
    #     0: random.choices(data_ctr[0], k=len(balance_data_2)),
    #     1: balance_data_mov_1 + balance_data_imm_1,
    #     2: balance_data_2,
    # }

    data = {
        0: data_ctr[0],
        1: data_ctr[1],
        2: data_ctr[2],
        'mv': mv_files,
        'imm': imm_files,
    }

    # count = min(len(data_ctr[0]), len(data_ctr[1]), len(data_ctr[2]))
    # data = {
    #     0: random.sample(data_ctr[0], count//2),
    #     1: random.sample(data_ctr[1], count),
    #     2: random.sample(data_ctr[2], count),
    # }

    with open(dest_path, 'wb') as f:
        pickle.dump(data, f)

def main(all_files, dest_path):
    num_workers = 30
    # all_files = glob.glob(str(data_path/'*/*.npz'))
    print(len(all_files))

    files_per_worker = max(1, int(len(all_files)/num_workers))
    
    workers = []
    # each worker gets a subset of all_files
    dest_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_workers):
        final_dest = f'{str(dest_path)}/{i}.pkl'

        start_idx = i*files_per_worker
        end_idx = (i+1)*files_per_worker
        # if i == num_workers-1:
        #     end_idx = len(all_files)
        files = all_files[start_idx:end_idx]

        workers.append(mp.Process(target=balance_data, args=(files, final_dest)))

    for worker in workers:
        worker.daemon = True
        worker.start()
    
    for worker in workers:
        worker.join()


if __name__ == '__main__':

    # files = glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/aug27/2_obs/1/*/*.npz')
    # files = glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/aug29/2_obs/0/*/*.npz') +  glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/aug27/2_obs/1/*/*.npz')
    # files = glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/aug29/2_obs/0/*/*.npz') +  glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/aug27/2_obs/1/*/*.npz')
    id = 1
    save_id = 0

    main_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_0'
    data_folder = f'{main_folder}/all_data'
    tmp_folder = f'{main_folder}/tmp'
    balance_folder = f'{main_folder}/balanced'


    # save_id = 2 -> has only 0 and 1 obs data
    # save_id = 3 -> has only 0, 1, 2 (unbalanced 2) obs data

    # files = glob(f'/common/users/dm1487/legged_manipulation_data_store/trajectories/sep16/2_obs/{id}/*/*.npz') 
    files = glob(f'{data_folder}/*/*.npz') 
    print(len(files))
    random.shuffle(files)

    dest_path = Path(f'{tmp_folder}/{save_id}')
    main(files[:], dest_path)

    combine_files = sorted(glob(f'{str(dest_path)}/*.pkl'))

    all_files = {
        0: [],
        1: [],
        2: [], 
        # 'mv': [],
        # 'imm': [],
    }
    for file in combine_files[:]:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            all_files[0] += data[0]
            all_files[1] += data[1]
            all_files[2] += data[2]
            all_files['mv'] += data['mv']
            all_files['imm'] += data['imm']
    
    min_len = min(len(all_files['mv']), len(all_files['imm']))
    # data_len = min(len(all_files[0]), len(all_files[1]), len(all_files[2]))
    for key in all_files.keys():
        print(key, len(all_files[key]))

    if not os.path.exists(balance_folder):
        os.makedirs(balance_folder)
    
    for key in all_files.keys():
        with open(f'{balance_folder}/train_{key}.pkl', 'wb') as f:
            if key == 'mv' or key == 'imm':
                pickle.dump(all_files[key][:min_len], f)
            else:
                pickle.dump(all_files[key][:], f)

    shutil.rmtree(tmp_folder)

    