import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt


def build_traj(data_path, dest_path):
    ctr = 0
    for f in tqdm(sorted(glob.glob(data_path+'/*.npz'))[:]):
        try:
            data = np.load(f)
            # print(data['input'].shape[0])
            for i in range(data['input'].shape[0]):
                new_data = {
                    'input': data['input'][i],
                    'target': data['target'][i],
                    'actions': data['actions'][i],
                    'fsw': data['fsw'][i],
                    'done': data['done'][i],
                }
            
                np.savez_compressed(f'{dest_path}/data_{ctr}.npz', **new_data)
                ctr += 1
        except:
            print(f'missed {ctr}')
            ctr += 1

def main(data_path, dest_path):
    workers = []
    for folders in tqdm(sorted(glob.glob(str(data_path/'*')))):
        final_dest = Path(dest_path)/str((Path(folders).stem))
        final_dest.mkdir(parents=True, exist_ok=True)
        workers.append(mp.Process(target=build_traj, args=(folders, final_dest)))

    for worker in workers:
        worker.start()
    
    for worker in workers:
        worker.join()

        # for f in tqdm(sorted(glob.glob(folders+'/*.npz'))[:-1]):
        #     data = np.load(f)
        #     for i in range(data['input'].shape[0]):
        #         new_data = {
        #             'input': data['input'][i],
        #             'target': data['target'][i],
        #             'actions': data['actions'][i],
        #             'fsw': data['fsw'][i],
        #             'done': data['done'][i],
        #         }
                
        #         np.savez_compressed(f'{final_dest}/data_{ctr}.npz', **new_data)
        #         ctr += 1
if __name__ == '__main__':
    data_path = Path(f'/common/users/dm1487/legged_manipulation_data/rollout_data/multi_policy_2/')
    dest_path = Path(f'/common/users/dm1487/legged_manipulation_data/rollout_data/multi_policy_2_single_trajectories')
    dest_path.mkdir(parents=True, exist_ok=True)
    main(data_path, dest_path)