import torch
import numpy as np
import pickle
from pathlib import Path
from matplotlib import patches as pch
from matplotlib import pyplot as plt
from matplotlib import animation
from indep_model.config import *
FFwriter = animation.FFMpegWriter

def get_visualization(idx, obs, priv_obs, pred_obs, pred, fsw):

    patch_set = []

    pos_rob, angle_rob = obs[idx, :2], torch.rad2deg(obs[idx, 3:4])
    pred_pos_rob, pred_angle_rob = pred_obs[idx, :2], torch.rad2deg(pred_obs[idx, 3:4])
    # angle_rob = torch.rad2deg(torch.atan2(2.0*(rot_rob[0]*rot_rob[1] + rot_rob[3]*rot_rob[2]), 1. - 2.*(rot_rob[1]*rot_rob[1] + rot_rob[2]*rot_rob[2])))

    for t_idx in range(4):
        if (t_idx == 2) or (t_idx == 0):
            patch_set.append(pch.Rectangle(pred_pos_rob.cpu().numpy() - np.array([0.588/2, 0.22/2]), width=0.588, height=0.22-0.05, angle=pred_angle_rob.cpu(), rotation_point='center', facecolor='black', label='pred_robot', alpha=0.5))
            if t_idx == 2:
                continue
        patch_set.append(pch.Rectangle(pos_rob.cpu().numpy() - np.array([0.588/2, 0.22/2]), width=0.588, height=0.22-0.05, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))
                        
    for i in range(RECTS):
        j = i*7 + 2
        
        contact = torch.sigmoid(priv_obs[idx][j-2])
        alpha = 0.3 if contact.item() < 0.5 else 0.8
        movable = torch.sigmoid(priv_obs[idx][j-1])

        pos, pos_pred, pos_fsw = priv_obs[idx][j:j+2], pred[idx][j:j+2], fsw[idx][j:j+2]
        angle, angle_pred, angle_fsw = torch.rad2deg(priv_obs[idx][j+2:j+3]), torch.rad2deg(pred[idx][j+2:j+3]), torch.rad2deg(fsw[idx][j+2:j+3])
        size, size_pred, size_fsw = priv_obs[idx][j+3:j+5], pred[idx][j+3:j+5], fsw[idx][j+3:j+5]

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
            patch_set.append(pch.Rectangle(pos.cpu() - size.cpu()/2, *(size.cpu()), angle=angle.cpu(), rotation_point='center', facecolor=block_color, label=f'true_mov_{i}'))
            patch_set.append(pch.Rectangle(pos_pred.cpu() - size_pred.cpu()/2, *(size_pred.cpu()), angle=angle_pred.cpu(), rotation_point='center', facecolor=pred_block_color, alpha=alpha, label=f'pred_mov_{i}'))
            patch_set.append(pch.Rectangle(pos_fsw.cpu() - size_fsw.cpu()/2, *(size_fsw.cpu()), angle=angle_fsw.cpu(), rotation_point='center', facecolor=block_color_fsw, label=f'fsw_mov_{i}'))

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