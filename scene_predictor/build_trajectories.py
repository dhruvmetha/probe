import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt


def build_traj(files, dest_path):
    ctr = 0
    for f in tqdm(sorted(files)[:]):
        try:
            data = np.load(f)
            print(data['input'].shape[0])
            for i in tqdm(range(data['input'].shape[0])):
                new_data = {
                    'input': data['input'][i],
                    'll_actions': data['ll_actions'][i],
                    'actions': data['actions'][i],
                    'target': data['target'][i],
                    'target_env': data['target_env'][i],
                    'fsw': data['fsw'][i],
                    'done': data['done'][i],

                    # 'input': self.input_data[env_ids].clone().cpu(), # dof_pos, dof_vel, torques_applied
                    # 'll_actions': self.ll_actions_data[env_ids].clone().cpu(), # ll_actions
                    # 'actions': self.actions_data[env_ids].clone().cpu(), # actions
                    # 'target': self.target_data[env_ids].clone().cpu(), # base_pos, base_ang, base_lin_vel, base_ang_vel
                    # 'target_env': self.target_env_data[env_ids].clone().cpu(), 
                    # 'fsw': self.fsw_data[env_ids].clone().cpu(),
                    # 'done': self.done_data[env_ids].clone().cpu(),
                }
            
                np.savez_compressed(f'{dest_path}/data_{ctr}.npz', **new_data)
                ctr += 1
        except:
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
    # id = 0
    # all_files = glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/final_illus/*/*.npz") # + glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/sep16/*lag_6*/*/*.npz")
    # all_files = glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/final_illus/*/*.npz") # + glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/sep16/*lag_6*/*/*.npz")
    # SEED = 20
    root_path = '/common/users/dm1487/legged_manipulation_data_store'
    root_traj_path = f'{root_path}/trajectories'
    sub_path = 'iros24_play_feb22/2_obs'
    all_files = glob.glob(f"{root_path}/{sub_path}/*/*/*.npz") # + glob.glob(f"/common/users/dm1487/legged_manipulation_data_store/2_obs/sep16/*lag_6*/*/*.npz")

    print(len(all_files))
    
    small_set = []
    small_set += all_files
    random.shuffle(small_set) 
    # sub_path = 'iros24_play_feb22/2_obs'
    dest_path = Path(f'{root_traj_path}/{sub_path}/all_data')
    dest_path.mkdir(parents=True, exist_ok=True)
    main(small_set, dest_path)