import torch
import numpy as np
import pickle
from pathlib import Path
from matplotlib import patches as pch
from matplotlib import pyplot as plt
from matplotlib import animation
# from indep_model.config import *
FFwriter = animation.FFMpegWriter
RECTS = 2


def get_real_visualization(obstacle_count,
                           robot_params,
                           gt_obstacle_params,
                           pred_obstacle_params):

    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    # # ax should have xlim, ylim, grid at 0.2 step for x and y
    # ax.set(xlim=(-1.0, 4.0), ylim=(-1, 1))
    # ax.axis('off')

    patches = []
    
    # robot patch
    rpos = np.array([robot_params[0], robot_params[1]])
    rrot = np.array(robot_params[2]) * 180/np.pi
    rsize = np.array([0.52, 0.30])

    patches.append(pch.Rectangle(rpos - rsize/2, *(rsize), angle=rrot, rotation_point='center', facecolor='green'))

    # grount truth obstacle patch
    k = 0
    for i in range(obstacle_count):
        obs_pos = np.array(gt_obstacle_params[k:k+2])
        obs_rot = np.array(gt_obstacle_params[k+2])  * 180/np.pi
        # obs_size = np.array(gt_obstacle_params[k+3:k+5])
        obs_size = np.array([0.43, 1.62]) if i == 0 else np.array([0.43, 0.62])
        if i == 0:
            facecolor = 'yellow'
        else:
            facecolor = 'red'
        patches.append(pch.Rectangle(obs_pos - obs_size/2, *(obs_size), angle=obs_rot, rotation_point='center', facecolor=facecolor))

        k += 5
    
    k = 2
    for i in range(obstacle_count):
        contact = torch.sigmoid(pred_obstacle_params[k-2])
        movable = torch.sigmoid(pred_obstacle_params[k-1])
        obs_pos = np.array(pred_obstacle_params[k:k+2])
        obs_rot = np.array(pred_obstacle_params[k+2])  * 180/np.pi
        obs_size = np.array(pred_obstacle_params[k+3:k+5])
        facecolor = 'blue'
        if torch.sigmoid(movable) > 0.5:
            facecolor = 'orange'
        patches.append(pch.Rectangle(obs_pos - obs_size/2, *(obs_size), angle=obs_rot, rotation_point='center', facecolor=facecolor))
        k += 7
    
    return patches


def get_visualization(idx, obs, priv_obs, pred_obs, pred, fsw, estimate_pose=False):

    obs = obs.cpu()
    pred_obs = pred_obs.cpu()
    pred = pred.cpu()
    priv_obs = priv_obs.cpu()

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
            fsw_j = i*7 + 3
            j = i*7 + 3
            
            confidence = torch.sigmoid(pred[idx][j-3]).item()
            contact = torch.sigmoid(pred[idx][j-2])
            # alpha = 0.3 if contact.item() < 0.5 else 0.8
            movable = torch.sigmoid(pred[idx][j-1])

            pos, pos_pred, pos_fsw = priv_obs[idx][j:j+2], pred[idx][j:j+2], fsw[idx][fsw_j:fsw_j+2]

            angle, angle_pred, angle_fsw = torch.rad2deg(priv_obs[idx][j+2:j+3]).numpy(), torch.rad2deg(pred[idx][j+2:j+3]).numpy(), torch.rad2deg(fsw[idx][fsw_j+2:fsw_j+3]).numpy()

            size, size_pred, size_fsw = priv_obs[idx][j+3:j+5], pred[idx][j+3:j+5], fsw[idx][fsw_j+3:fsw_j+5]

            pos, pos_pred, pos_fsw = pos.numpy(), pos_pred.numpy(), pos_fsw.numpy()
            size, size_pred, size_fsw = size.numpy(), size_pred.numpy(), size_fsw.numpy()

            block_color = 'red'
            if priv_obs[idx][j-1] == 1:
                block_color = 'yellow'

            block_color_fsw = 'red'
            if fsw[idx][fsw_j-1] == 1:
                block_color_fsw = 'yellow'
            
            pred_block_color = 'blue'
            if movable > 0.5:
                pred_block_color = 'orange'

            for _ in range(2):
                patch_set.append(pch.Rectangle(pos - size/2, *(size), angle=angle, rotation_point='center', facecolor=block_color, label=f'true_mov_{i}'))

                if np.prod(size_pred) > 0.05 or confidence > 0.5:
                    patch_set.append(pch.Rectangle(pos_pred - size_pred/2, *(size_pred), angle=angle_pred, rotation_point='center', facecolor=pred_block_color,  label=f'pred_mov_{i}'))
                else:
                    patch_set.append(pch.Rectangle(pos_pred - size_pred/2, *([0., 0.]), angle=angle_pred, rotation_point='center', facecolor=pred_block_color,  label=f'pred_mov_{i}'))
                
                patch_set.append(pch.Rectangle(pos_fsw - size_fsw/2, *(size_fsw), angle=angle_fsw, rotation_point='center', facecolor=block_color_fsw, label=f'fsw_mov_{i}'))
                

    return patch_set


def get_animation(patches):

    # file_name = Path(tmp_img_path).stem
    # # if os.path.exists(f"{dest_folder}/{file_name}.mp4"):
    # #     continue
    fig, axes = plt.subplots(2, 2, figsize=(24, 24))
    ax = axes.flatten()

    # try:
    #     with open(tmp_img_path, 'rb') as f:
    #         patches = pickle.load(f)
    # except:
    #     plt.close()
    #     return False
    
    last_patch = []

    def animate(frame):
        if len(last_patch) != 0:
            for i in last_patch:
                try:
                    i.remove()
                except:
                    pass
            last_patch.clear()
        
        robot, robot_1, robot_2, robot_3 = frame[0], frame[1], frame[2], frame[3]

        ax[0].add_patch(robot)
        ax[1].add_patch(robot_1)
        ax[2].add_patch(robot_2)
        ax[3].add_patch(robot_3)

        ax[0].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='all')
        ax[1].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='truth')
        ax[2].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='predicted')
        ax[3].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='full seen world')
        
        for i in range(RECTS):
            j = i*6 + 4
            ax[0].add_patch(frame[j])
            ax[0].add_patch(frame[j+1])

            ax[1].add_patch(frame[j+3])
            ax[2].add_patch(frame[j+4])

            ax[3].add_patch(frame[j+5])

        last_patch.extend(frame)
    
    anim = animation.FuncAnimation(fig, animate, frames=patches, interval=10, repeat=False)
    plt.close()
    return anim