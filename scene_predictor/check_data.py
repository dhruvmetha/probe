from pathlib import Path
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import torch
import pickle

if __name__ == '__main__':

    # data_folder ='/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep16/2_obs/all_files_bal/all_files_2_train.pkl'

    # with open(data_folder, 'rb') as f:
    #     balanced_data = pickle.load(f)
    # # balanced_data = glob(data_folder)
    # random.shuffle(balanced_data)
    # print('# trajectories:', len(balanced_data))

    # # files = glob(str(dest_path1/'*/*/*.npz')) 
    # print(len(balanced_data))

    files = glob(f'/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24/all_data/*/*.npz')

    # print(len(files))
    # exit()

    xs, ys = [], []
    while True:
        lin_vel = []
        plt.figure(figsize=(20, 8))
        # plt.xlim(-1.0, 4.0)
        # plt.ylim(-1.0, 1.0)
        plt.grid()
        # print(len(glob(str(dest_path/'*/*.npz'))))
        # files = glob(str(dest_path/'*/*.npz')) # [:10000]  + glob(str(dest_path1/'*/*.npz'))[:10000] + glob(str(dest_path2/'*/*.npz'))[:10000] + glob(str(dest_path3/'*/*.npz'))[:10000]
        # + glob(str(dest_path2/'*/*.npz')) + glob(str(dest_path3/'*/*.npz'))# + glob(str(dest_path2/'*/*.npz'))[:10000] + glob(str(dest_path3/'*/*.npz'))[:10000]
        
        # exit()
        points = 0
        total_points = 1
        while True:
            if points == total_points:
                break
            # exit()
            print(f'{points}/{total_points}', end='\r')
            file = random.choice(files)
            data = np.load(file)
            # actions = data['actions']
            # lin_vel.extend((np.clip(actions[:, 0], -0.65, 0.65)).tolist())
            dones = data['done']

            last_idx = dones.nonzero()[0][-1]

            print(data['target'][:last_idx])
            pose = data['target'][:last_idx, :2]

            plt.plot(pose[:, 0] * 4, pose[:, 1])
            plt.show()
            continue
            for k in range(0, min(100, last_idx)):
                # if k == 100:
                #     break
                
                target_env = data['target_env']
                
                t_k = target_env[k]
                last = target_env[last_idx]
                x = t_k[2]
                y = t_k[3]

                x_1 = last[9]
                x_2 = last[16]

                if x > 0.0:
                    xs.append(x)
                    ys.append(y)
                    points += 1
                    break
                    

            # print(last_idx)
            # if len(last_idx) > 0:
            #     last_idx = last_idx[0]
            # else:
            #     # print(len(dones))
            #     last_idx = len(dones)
            # # print(last_idx)

            # pos = data['target'][1:last_idx, :2]
        # plt.scatter(xs, ys)
        plt.hist(xs, bins=20)
        # print(len(lin_vel))
        # plt.scatter(np.arange(0, len(lin_vel)), lin_vel)
        plt.show()