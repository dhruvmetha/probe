from matplotlib import pyplot as plt
from matplotlib import patches as pch
from tqdm import tqdm
from glob import glob
from pathlib import Path
import os
import random
import numpy as np
import multiprocessing as mp

def build_images(files, final_dest):
    m = 0
    for file in tqdm(files):
        dest = f'{final_dest}/{m}'
        m += 1
        if not os.path.exists(dest):
            os.makedirs(dest)
        data = np.load(file)
        done_idx = data['done'].nonzero()[0]
        target = data['target_env']
        t = target[done_idx[-1], :]

        for d_idx in done_idx:
            t = target[d_idx, :]
            k = 0
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
            ax.set(xlim=(-1.0, 4.0), ylim=(-1, 1))
            # no axis
            ax.axis('off')
            patches = []
            for i in range(3):
                color = 'red' if t[k+1] == 0 else 'yellow'
                x, y, theta = t[k+2], t[k+3], t[k+4]
                w, h = t[k+5], t[k+6]
                k += 7
                pos = np.array([x, y])
                size = np.array([w, h])
                angle = np.array([theta * 180 / np.pi])
                if np.sum(size) < 0.1:
                    continue
                patches.append(pch.Rectangle(pos - size/2, *(size), angle=angle, rotation_point='center', facecolor=color))
            if len(patches) > 0:
                for patch in patches:
                    ax.add_patch(patch)
                fig.savefig(f'{dest}/{d_idx}.png')
            plt.close()

def main(all_files, dest_path):
    num_workers = 24
    # all_files = glob.glob(str(data_path/'*/*.npz'))
    print(len(all_files))

    files_per_worker = max(1, int(len(all_files)/num_workers))
    
    workers = []
    # each worker gets a subset of all_files
    dest_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_workers):
        final_dest = f'{str(dest_path)}/{i}'

        start_idx = i*files_per_worker
        end_idx = (i+1)*files_per_worker
        # if i == num_workers-1:
        #     end_idx = len(all_files)
        files = all_files[start_idx:end_idx]

        workers.append(mp.Process(target=build_images, args=(files, final_dest)))

    for worker in workers:
        worker.daemon = True
        worker.start()
    
    for worker in workers:
        worker.join()
if __name__ == "__main__":

    files = glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/2_obs/12/*/*.npz')
    print(len(files))

    main(files[:10000], Path('/common/users/dm1487/legged_manipulation_data_store/images/2_obs/12'))

    
