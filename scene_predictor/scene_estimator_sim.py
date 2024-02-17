from scene_estimator import SceneEstimator
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import pickle
import numpy as np
import torch
import os
from data import SceneEstimatorDataset
from torch.utils.data import DataLoader
import random
from glob import glob


FFwriter = animation.FFMpegWriter

HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/2023-09-22_00-22-04'
DATA_DIR = '/common/users/dm1487/legged_manipulation_data_store/trajectories/sep16/2_obs_final_illus'
RECTS = 3

def create_animation(patch_set, save_to=None, filename=None):
    fig, ax = plt.subplots(1, 1, figsize=(12.80, 7.20), dpi=100)
    ax.axis('off')
    ax.set(xlim=(-1.0, 4.0), ylim=(-1, 1), aspect='auto')
    last_patch = []

    def animate(frame):
        if len(last_patch) != 0:
            for i in last_patch:
                try:
                    i.remove()
                except:
                    pass
            last_patch.clear()
        
        for patch in frame:
            if patch is not None:
                ax.add_patch(patch)
                last_patch.append(patch)

    anim = animation.FuncAnimation(fig, animate, frames=patch_set, interval=10, repeat=False)
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    anim.save(f"{save_to}/{filename}.mp4", writer = FFwriter(10))
    plt.close()


if __name__ == '__main__':
    device = 'cuda:0'
    num_layers = 2
    num_heads = 2
    hidden_size = 1024
    embed_size = 512
    se = SceneEstimator(input_size=42, num_layers=num_layers, num_heads=num_heads, hidden_size=hidden_size, embed_size=embed_size, causal=True, device=device, start=0)
    model_id = 126
    model_path = f'{HOME_DIR}/checkpoints/model_{model_id}.pt'
    se.load_model(model_path, device=device)

    data_files = glob(f'{DATA_DIR}/*/*/*.npz')

    sub_val_files = data_files # random.sample(data_files, 20000)
    val_ds = SceneEstimatorDataset(files=sub_val_files, sequence_length=1500, estimate_pose=True, start=0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

    for idx, (inp, target, mask, fsw, pose) in tqdm(enumerate(val_dl)):
        inp = inp.to(device)
        # target = target
        out = se.predict(inp).cpu()

        target *= torch.tensor([1, 1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3)
        out *= torch.tensor([1, 1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3)

        done_idx = mask.view(1500).nonzero()[-1]

        target = target.numpy()
        out = out.numpy()
        pose = pose.numpy()
        
        patch_set = []
        for i in range(done_idx):
            patches = []
            pos = pose[0, i, :2] * np.array([1/0.25, 1])
            ang = (pose[0, i, 2] * 3.14) * 180/np.pi
            size = np.array([0.52, 0.30])

            patches.append(pch.Rectangle(pos - size/2, *(size), angle=ang, rotation_point='center', facecolor='green'))

            for j in range(2):
                k = j*8 + 3
                pos = target[0, i, k:k+2]
                ang = target[0, i, k+2] * 180/np.pi
                size = target[0, i, k+3:k+5]

                color = 'yellow' if j == 0 else 'red'

                if np.prod(size) > 0.1:
                    patches.append(pch.Rectangle(pos - size/2, *(size), angle=ang, rotation_point='center', facecolor=color))
                else:
                    patches.append(None)
            sigmoid = lambda z: 1/(1 + np.exp(-z))
            for j in range(2):
                conf = 1 if sigmoid(out[0, i, j*8]) > 0.8 else 0
                k = j*8 + 3
                pos = out[0, i, k:k+2]
                ang = out[0, i, k+2] * 180/np.pi
                size = out[0, i, k+3:k+5]

                color = 'orange' if j == 0 else 'blue'

                if np.prod(size) > 0.1:
                    patches.append(pch.Rectangle(pos - size/2, *(size), angle=ang, rotation_point='center', facecolor=color,alpha = conf))
                else:
                    patches.append(None)
                    
            patch_set.append(patches)
        
        create_animation(patch_set, save_to=f'{HOME_DIR}/final_animations1_{model_id}', filename=f'{idx}')

            


        # save_eval = {
        #     'pred' : torch.cat([out[:, :, 3:8], out[:, :, 11:16]], dim=-1).numpy(),
        #     'gt' : torch.cat([target[:, :, 3:8], target[:, :, 11:16]], dim=-1).numpy(), 
        #     'mask': mask.numpy()
        # }


        

        # with open(f'/common/users/dm1487/legged_manipulation_data_store/evaluation_data/{idx}.pkl' , 'wb') as f:
        #     pickle.dump(save_eval, f)

        # print(save_eval['pred'].shape, save_eval['gt'].shape, save_eval['mask'].shape)

        # print(out.shape, fsw.shape, target.shape)

        # patch_set = []
        # for i in range(done_idx):
        #     patch_set.append(get_visualization(0, target[:, i, :], fsw[:, i, :], target[:, i, :], out[:, i, :] * torch.tensor([1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3), fsw[:, i, :], estimate_pose=False))
        # create_animation(patch_set, save_to=f'{HOME_DIR}/2023-08-31_08-10-20/test_animations_{id}', filename=f'{idx}')
        # print(f'{idx} ready to view')


