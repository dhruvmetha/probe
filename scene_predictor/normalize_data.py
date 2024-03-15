from pathlib import Path
from glob import glob

import numpy as np
from tqdm import tqdm

import pickle
import random
import os
import multiprocessing as mp
import shutil
        

def calculate_box_corners(x, y, theta, w, h):
    """
    Calculate the bottom-left and top-right corners of a 2D box given its center,
    width, height, and rotation angle.
    
    Parameters:
    - x, y: Coordinates of the box's center.
    - theta: Rotation angle of the box in radians.
    - w, h: Width and height of the box.
    
    Returns:
    - (bl_x, bl_y, tr_x, tr_y): Coordinates of the bottom-left and top-right corners.
    """
    def rotate_point(px, py, theta):
        """Rotate a point by theta around the origin."""
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_rotated = px * cos_theta - py * sin_theta
        y_rotated = px * sin_theta + py * cos_theta
        return x_rotated, y_rotated

    # Half dimensions
    half_w, half_h = w / 2, h / 2
    
    # Calculate offsets for corners in local box coordinates
    offsets = [(-half_w, -half_h), (half_w, half_h)]
    
    # Rotate offsets and translate to world coordinates
    corners = [rotate_point(ox, oy, theta) for ox, oy in offsets]
    corners_world = [(x + cx, y + cy) for cx, cy in corners]
    
    # Bottom-left and top-right corners
    bl_x, bl_y = corners_world[0]
    tr_x, tr_y = corners_world[1]
    
    return bl_x, bl_y, tr_x, tr_y

def balance_data(files, dest_path):
    ctr = 0
    data_ctr = {
        0: [],
        1: [],
        2: [], 
        3: [],
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
            last_idx = min(500, data['done'].nonzero()[0][-1])
            target = data['target_env']

            # np.zeros(((last_idx, 8)))
            # x = target[100, 2]
            # y = target[100, 3]
            # theta = target[100, 4]
            # w = target[100, 5]
            # h = target[100, 6]

            # print(x, y, theta, w, h)
            # print(calculate_box_corners(x, y, theta, w, h))
            
            # x = target[100, 11]
            # y = target[100, 12]
            # theta = target[100, 13]
            # w = target[100, 14]
            # h = target[100, 15]

            # print(x, y, theta, w, h)
            # print(calculate_box_corners(x, y, theta, w, h))

            index_found = int(int(target[last_idx, 2] != 0) + int(target[last_idx, 11] != 0) + int(target[last_idx, 20]!= 0))
            # print(index_found)
            data_ctr[index_found].append(file)

            # if index_found == 1:
            #     if int(target[last_idx, 2] != 0):
            #         mv_files.append(file)
            #         if target[last_idx, 3] > 0:
            #             # right side immovable
            #             data_1['mov']['left'].append(file)
            #         else:
            #             # left side immovable
            #             data_1['mov']['right'].append(file)
            #     else:
            #         imm_files.append(file)
            #         if target[last_idx, 12] > 0:
            #             # right side immovable
            #             data_1['imm']['left'].append(file)
            #         else:
            #             # left side immovable
            #             data_1['imm']['right'].append(file)

            # if index_found == 2:
            #     if target[last_idx, 3] > 0:
            #         if target[last_idx, 12] > 0:
            #             # right side immovable
            #             data_2['left']['left'].append(file)
            #         else:
            #             # left side immovable
            #             data_2['left']['right'].append(file)
            #     else:
            #         if target[last_idx, 12] > 0:
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
            
    # print(len(data_ctr[0]))

    # print(len(data_1['mov']['left']), len(data_1['mov']['right']), len(data_1['imm']['left']), len(data_1['imm']['right']))

    # print(len(data_2['left']['left']), len(data_2['left']['right']), len(data_2['right']['left']), len(data_2['right']['right']))

    min_count = min(min(len(data_2['left']['left']), len(data_2['left']['right']), len(data_2['right']['left']), len(data_2['right']['right'])), min(len(data_1['mov']['left']), len(data_1['mov']['right']), len(data_1['imm']['left']), len(data_1['imm']['right'])))

    # print(len(data_ctr[0]), min_count)

    # d = random.sample(data_2['left']['left'], min_count) + random.sample(data_2['left']['right'], min_count) + random.sample(data_2['right']['left'], min_count) + random.sample(data_2['right']['right'], min_count) + random.sample(data_1['mov']['left'], min_count) + random.sample(data_1['mov']['right'], min_count) + random.sample(data_1['imm']['left'], min_count) + random.sample(data_1['imm']['right'], min_count)

    # print(len(d), len(data_ctr[0]))



    # count = min(len(data_1['imm']['left']), len(data_1['imm']['right']), len(data_1['mov']['left']), len(data_1['mov']['right'])) # , len(data_2['right']['left']), len(data_2['right']['right']), len(data_2['left']['left']), len(data_2['left']['right']))
    
    # balance_data_mov_1 = random.sample(data_1['mov']['left'], min(count*2, len(data_1['mov']['left']))) + random.sample(data_1['mov']['right'], min(count*2, len(data_1['mov']['right'])))

    # balance_data_imm_1 = random.sample(data_1['imm']['left'], min(count*2, len(data_1['imm']['left']))) + random.sample(data_1['imm']['right'], min(count*2, len(data_1['imm']['right'])))
    
    # balance_data_2 = random.sample(data_2['left']['left'], min(count, len(data_2['left']['left']))) + random.sample(data_2['left']['right'], min(count, len(data_2['left']['right']))) + random.sample(data_2['right']['left'], min(count, len(data_2['right']['left']))) + random.sample(data_2['right']['right'], min(count, len(data_2['right']['right'])))
    
    
    # data = {
    #     0: random.choices(data_ctr[0], k=len(balance_data_2)),
    #     1: balance_data_mov_1 + balance_data_imm_1,
    #     2: balance_data_2,
    # }

    data_0_consolidated = data_ctr[0]  # random.sample(data_ctr[0], min_count)
    data_1_consolidated = data_ctr[1] # random.sample(data_1['mov']['left'], min_count) + random.sample(data_1['mov']['right'], min_count) + random.sample(data_1['imm']['left'], min_count) + random.sample(data_1['imm']['right'], min_count)
    data_2_consolidated = data_ctr[2] # random.sample(data_2['left']['left'], min_count) + random.sample(data_2['left']['right'], min_count) + random.sample(data_2['right']['left'], min_count) + random.sample(data_2['right']['right'], min_count)
    data_3_consolidated = data_ctr[3]
    data = {
        0: data_0_consolidated,
        1: data_1_consolidated,
        2: data_2_consolidated,
        3: data_3_consolidated,
        # 'mv': mv_files,
        # 'imm': imm_files,
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
    num_workers = 40
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

    root_path = '/common/users/dm1487/legged_manipulation_data_store'
    root_traj_path = f'{root_path}/trajectories'
    sub_path = 'iros24_play_mar14/3_obs'

    main_folder = f'{root_traj_path}/{sub_path}'
    data_folder = f'{main_folder}/all_data'
    tmp_folder = f'{main_folder}/tmp'
    balance_folder = f'{main_folder}/no_balanced_test'


    # save_id = 2 -> has only 0 and 1 obs data
    # save_id = 3 -> has only 0, 1, 2 (unbalanced 2) obs data

    # files = glob(f'/common/users/dm1487/legged_manipulation_data_store/trajectories/sep16/2_obs/{id}/*/*.npz') 
    files = glob(f'{data_folder}/*/*.npz') 
    print(len(files))
    random.shuffle(files)

    dest_path = Path(f'{tmp_folder}/{save_id}')
    main(files, dest_path)

    combine_files = sorted(glob(f'{str(dest_path)}/*.pkl'))

    all_files = {
        0: [],
        1: [],
        2: [], 
        3: [],
        'mv': [],
        'imm': [],
    }
    for file in combine_files[:]:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            all_files[0] += data[0]
            all_files[1] += data[1]
            all_files[2] += data[2]
            all_files[3] += data[3]
            # all_files['mv'] += data['mv']
            # all_files['imm'] += data['imm']
    
    data_len = min(len(all_files[0]), len(all_files[1]), len(all_files[2]))

    print(data_len)
    for key in all_files.keys():
        print(key, len(all_files[key]))

    if not os.path.exists(balance_folder):
        os.makedirs(balance_folder)
    
    for key in all_files.keys():
        with open(f'{balance_folder}/train_{key}.pkl', 'wb') as f:
            # if key == 'mv' or key == 'imm':
            #     random.shuffle(all_files[key])
            #     pickle.dump(all_files[key][:data_len], f)
            # else:
            pickle.dump(all_files[key], f)

    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    