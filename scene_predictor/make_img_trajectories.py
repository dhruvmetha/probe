import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches as pch
from PIL import Image

def build_traj(files, dest_path):
    ctr = 0
    for f in tqdm(sorted(files)[:]):
        try:
            data = np.load(f)
            # print(data['input'].shape[0])
            # for i in tqdm(range(data['input'].shape[0])):
            new_data = {
                'input': data['input'],
                # 'll_actions': data['ll_actions'],
                # 'actions': data['actions'],
                'target': data['target'],
                'target_env': data['target_env'],
                'fsw': data['fsw'],
                'done': data['done'],
            }

            target_env = data['target_env']
            done_idx = data['done'].nonzero()[0][-1]
            obs_img = np.zeros((done_idx, 150, 300, 3))
            fig, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=100)
            for j in range(done_idx):
                plt.cla()
                x, y, angle, size_x, size_y = target_env[j, 2:7]
                angle = np.rad2deg(angle)
                rect = pch.Rectangle(np.array([x, y]) - np.array([size_x, size_y])/2, size_x, size_y, angle=angle, rotation_point='center', facecolor='blue')
                ax.add_patch(rect)

                x, y, angle, size_x, size_y = target_env[j, 11:16]
                angle = np.rad2deg(angle)
                rect = pch.Rectangle(np.array([x, y]) - np.array([size_x, size_y])/2, size_x, size_y, angle=angle, rotation_point='center', facecolor='red')
                ax.add_patch(rect)

                # set the axis limits
                ax.set(xlim=(0.0, 4.0), ylim=(-1, 1))
                ax.axis('off')

                plt.savefig(f'{dest_path}/tmp.png', format='png', )
                img = Image.open(f'{dest_path}/tmp.png')
                image_array = np.array(img)
                obs_img[j] = image_array[:, :, :3]
            plt.close()
            new_data['obs_img'] = obs_img       
            np.savez_compressed(f'{dest_path}/data_{ctr}.npz', **new_data)
            ctr += 1
        except Exception as e:
            print(e)
            print(f'missed {ctr}')
            ctr += 1

def main(all_files, dest_path):
    num_workers = 40
    # all_files = glob.glob(str(data_path/'*/*.npz'))
    print(len(all_files))

    files_per_worker = max(1, int(len(all_files)/num_workers))

    workers = []
    # each worker gets a subset of all_files
    for i in range(num_workers):
        final_dest = Path(dest_path)/str(i)
        final_dest.mkdir(parents=True, exist_ok=True)

        start_idx = i*files_per_worker
        end_idx = (i+1)*files_per_worker
        # if i == num_workers-1:
        #     end_idx = len(all_files)
        files = all_files[start_idx:end_idx]

        workers.append(mp.Process(target=build_traj, args=(files, final_dest)))
                     
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
    import random
    import pickle
    # id = 0
    # all_files = glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/final_illus/*/*.npz") # + glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/sep16/*lag_6*/*/*.npz")
    # all_files = glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/final_illus/*/*.npz") # + glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/sep16/*lag_6*/*/*.npz")
    # SEED = 20
    root_path = '/common/users/dm1487/legged_manipulation_data_store'
    root_traj_path = f'{root_path}/trajectories'
    sub_path = 'iros24_play_feb21/2_obs'
    files = ['train_0.pkl', 'train_1.pkl', 'train_2.pkl']
    all_files = []
    for file in files:
        with open(f'{root_traj_path}/{sub_path}/balanced/{file}', 'rb') as f:
            all_files += pickle.load(f)

    # all_files = glob.glob(f"{root_traj_path}/{sub_path}/*/*.npz") # + glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/sep16/*lag_6*/*/*.npz")
    
    # print(len(all_files))
    # exit()

    small_set = []
    small_set += all_files
    random.shuffle(small_set) 
    sub_path = 'iros24_img_play_feb21/2_obs'
    dest_path = Path(f'{root_traj_path}/{sub_path}/all_data')
    dest_path.mkdir(parents=True, exist_ok=True)
    main(small_set, dest_path)