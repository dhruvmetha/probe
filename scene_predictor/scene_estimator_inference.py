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


FFwriter = animation.FFMpegWriter

HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-16_17-34-37/'
DATA_DIR = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep16/2_obs/all_files_bal/all_files_1_val.pkl'
RECTS = 3

def get_visualization(idx, obs, priv_obs, pred_obs, pred, fsw, estimate_pose=False):

    patch_set = []
    
    pos_rob, angle_rob = obs[idx, :2].numpy(), torch.rad2deg(obs[idx, 2:3]).numpy()
    pred_pos_rob, pred_angle_rob = pred_obs[idx, :2].numpy(), torch.rad2deg(pred_obs[idx, 2:3]).numpy()

    for t_idx in range(4):
        if (t_idx == 2) or (t_idx == 0):
            patch_set.append(pch.Rectangle(pred_pos_rob - np.array([0.588/2, 0.22/2]), width=0.588, height=0.22-0.05, angle=pred_angle_rob, rotation_point='center', facecolor='black', label='pred_robot', alpha=0.5))
            if t_idx == 2:
                continue
        patch_set.append(pch.Rectangle(pos_rob - np.array([0.588/2, 0.22/2]), width=0.588, height=0.22-0.05, angle=angle_rob, rotation_point='center', facecolor='green', label='robot'))
                        

    # print(estimate_pose)
    if not estimate_pose:
        for i in range(RECTS):
            j = i*7 + 2
            
            # contact = torch.sigmoid(priv_obs[idx][j-2])
            # alpha = 0.3 if contact.item() < 0.5 else 0.8
            movable = torch.sigmoid(priv_obs[idx][j-1])

            pos, pos_pred, pos_fsw = priv_obs[idx][j:j+2], pred[idx][j:j+2], fsw[idx][j:j+2]

            angle, angle_pred, angle_fsw = torch.rad2deg(priv_obs[idx][j+2:j+3]).numpy(), torch.rad2deg(pred[idx][j+2:j+3]).numpy(), torch.rad2deg(fsw[idx][j+2:j+3]).numpy()

            size, size_pred, size_fsw = priv_obs[idx][j+3:j+5], pred[idx][j+3:j+5], fsw[idx][j+3:j+5]

            pos, pos_pred, pos_fsw = pos.numpy(), pos_pred.numpy(), pos_fsw.numpy()
            size, size_pred, size_fsw = size.numpy(), size_pred.numpy(), size_fsw.numpy()

            block_color = 'red'
            if priv_obs[idx][j-1] == 1:
                block_color = 'yellow'

            block_color_fsw = 'red'
            if fsw[idx][j-1] == 1:
                block_color_fsw = 'yellow'
            
            pred_block_color = 'blue'
            if movable > 0.5:
                pred_block_color = 'orange'

            for _ in range(2):
                patch_set.append(pch.Rectangle(pos - size/2, *(size), angle=angle, rotation_point='center', facecolor=block_color, label=f'true_mov_{i}'))
                if True or np.prod(size_pred) > 0.05:
                    patch_set.append(pch.Rectangle(pos_pred - size_pred/2, *(size_pred), angle=angle_pred, rotation_point='center', facecolor=pred_block_color,  label=f'pred_mov_{i}'))
                else:
                    patch_set.append(pch.Rectangle(pos_pred - size_pred/2, *([0., 0.]), angle=angle_pred, rotation_point='center', facecolor=pred_block_color,  label=f'pred_mov_{i}'))
                
                patch_set.append(pch.Rectangle(pos_fsw - size_fsw/2, *(size_fsw), angle=angle_fsw, rotation_point='center', facecolor=block_color_fsw, label=f'fsw_mov_{i}'))
                

    return patch_set

def create_animation(patch_set, save_to=None, filename=None):

    fig, axes = plt.subplots(2, 2, figsize=(48, 24))
    ax = axes.flatten()
    
    last_patch = []

    def animate(frame):

        if len(last_patch) != 0:
            for i in last_patch:
                try:
                    i.remove()
                except:
                    pass
            last_patch.clear()
        
        pred_robot, robot, robot_1, robot_2, robot_3 = frame[0], frame[1], frame[2], frame[3], frame[4]

        ax[0].add_patch(pred_robot)
        ax[0].add_patch(robot)
        ax[1].add_patch(robot_1)
        ax[2].add_patch(robot_2)
        ax[3].add_patch(robot_3)

        ax[0].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='all', aspect='auto')
        ax[1].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='truth', aspect='auto')
        ax[2].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='predicted', aspect='auto')
        ax[3].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='full seen world', aspect='auto')
        
    
        for i in range(RECTS):
            j = i*6 + 5
            ax[0].add_patch(frame[j])
            ax[0].add_patch(frame[j+1])

            ax[1].add_patch(frame[j+3])
            ax[2].add_patch(frame[j+4])

            ax[3].add_patch(frame[j+5])

        last_patch.extend(frame)
    
    anim = animation.FuncAnimation(fig, animate, frames=patch_set, interval=10, repeat=False)
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    anim.save(f"{save_to}/{filename}.mp4", writer = FFwriter(20))
    plt.close()

if __name__ == '__main__':
    device = 'cuda:0'
    num_layers = 2
    num_heads = 2
    hidden_size = 1024
    embed_size = 512
    se = SceneEstimator(input_size=30, num_layers=num_layers, num_heads=num_heads, hidden_size=hidden_size, embed_size=embed_size, causal=True, device=device, start=0)
    model_id = 21
    model_path = f'{HOME_DIR}/checkpoints/model_{model_id}.pt'
    se.load_model(model_path, device=device)

    with open(f'{DATA_DIR}', 'rb') as f:
        data_files = pickle.load(f)

    sub_val_files = data_files # random.sample(data_files, 20000)
    val_ds = SceneEstimatorDataset(files=sub_val_files, sequence_length=1500, estimate_pose=True, start=0)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=True)

    for idx, (inp, target, mask, fsw, pose) in tqdm(enumerate(val_dl)):
        inp = inp.to(device)
        target = target
        out = se.predict(inp).cpu()

        target *= torch.tensor([1, 1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3)
        out *= torch.tensor([1, 1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3)

        save_eval = {
            'pred' : torch.cat([out[:, :, 3:8], out[:, :, 11:16]], dim=-1).numpy(),
            'gt' : torch.cat([target[:, :, 3:8], target[:, :, 11:16]], dim=-1).numpy(), 
            'mask': mask.numpy()
        }

        if not os.path.exists(f'/common/users/dm1487/legged_manipulation_data_store/evaluation_data2'):
            os.makedirs(f'/common/users/dm1487/legged_manipulation_data_store/evaluation_data2')

        with open(f'/common/users/dm1487/legged_manipulation_data_store/evaluation_data2/{idx}.pkl' , 'wb') as f:
            pickle.dump(save_eval, f)

        # print(save_eval['pred'].shape, save_eval['gt'].shape, save_eval['mask'].shape)

        # print(out.shape, fsw.shape, target.shape)

        # patch_set = []
        # for i in range(done_idx):
        #     patch_set.append(get_visualization(0, target[:, i, :], fsw[:, i, :], target[:, i, :], out[:, i, :] * torch.tensor([1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3), fsw[:, i, :], estimate_pose=False))
        # create_animation(patch_set, save_to=f'{HOME_DIR}/2023-08-31_08-10-20/test_animations_{id}', filename=f'{idx}')
        # print(f'{idx} ready to view')


