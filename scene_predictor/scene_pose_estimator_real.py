from scene_pose_estimator import PoseSceneEstimator
from tqdm import tqdm
from glob import glob
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import pickle
import numpy as np
import torch
import os


FFwriter = animation.FFMpegWriter


HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-15_16-42-21'
DATA_DIR = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep1/2_obs/all_files'
RECTS = 3

def create_animation(patch_set, save_to=None, filename=None):

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set(xlim=(-1.0, 4.0), ylim=(-1, 1), xticks=np.arange(-1, 4.0, 0.2), yticks=np.arange(-1, 1.0, 0.2), title='all', aspect='auto')
    ax.grid()
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
    hidden_size = 2048
    embed_size = 512
    se = PoseSceneEstimator(input_size=42, num_layers=num_layers, num_heads=num_heads, hidden_size=2048, embed_size=512, causal=True, device=device)
    model_id = 21

    model_path = f'{HOME_DIR}/checkpoints/model_{model_id}.pt'
    se.load_model(model_path, device=device)

    files_list = []
    folders_list = []

    # files = list(range(11)); folder = 'sep4'
    # files_list.append(files); folders_list.append(folder)
    # files = [1, 2, 3, 5]; folder = 'sep3'
    # files_list.append(files); folders_list.append(folder)
    # files = [8, 10, 13, 14, 15, 20]; folder = 'sep2'
    # files_list.append(files); folders_list.append(folder)


    files = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; folder = 'sep15'
    files_list.append(files); folders_list.append(folder)

    files = [3, 4, 5]; folder = 'sep14'
    files_list.append(files); folders_list.append(folder)

    for files, folder in zip(files_list, folders_list):
        for file_no in files:
            # file_no = 8
            id = f'{file_no}'
            data_files = sorted(glob(f'/common/home/dm1487/Downloads/{folder}/{file_no}/*.npz'))[-1]

            data = np.load(data_files, allow_pickle=True)
            done_idx = data['done'].nonzero()[0][-1]
            start = 4
            end = done_idx
            inp = torch.from_numpy(data['input'][start:end]).unsqueeze(0)

            # setting up dof_pos, dof_vel, torque scaling
            inp[:, :, :3] *= 1
            inp[:, :, 3:15] *= 1
            inp[:, :, 15:27] *= 0.05
            # inp[:, :, 27:39] = torch.from_numpy(data['torques_estimated'][start:end]).unsqueeze(0)
            inp[:, :, 27:39] *= 0.08

            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 11]
            # idxes = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.long) + 3
            # inp[:, :, 3:15] = inp[:, :, idxes]
            # idxes = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.long) + 15
            # inp[:, :, 15:27] = inp[:, :, idxes]
            # idxes = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.long) + 27
            # inp[:, :, 27:39] = inp[:, :, idxes]

            # concatenating pose
            pose = (torch.from_numpy(data['target'][start:end]) * torch.tensor([0.25, 1, 1/3.14])).unsqueeze(0)
            inp = torch.cat([inp], dim=-1)

            inp = torch.cat([inp, torch.zeros(1, 1500 - (end-start), inp.shape[2])], dim=1) # padding

            m_obstacle_data = data['obstacles'][0]
            im_obstacle_data = data['obstacles'][2]
            rob_pos = data['target']

            new_pose, new_targ = se.predict(inp)
            new_pose, new_targ = new_pose.squeeze(0).cpu(), new_targ.cpu()
            # new_targ *= torch.tensor([1, 1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3).unsqueeze(0) 
            new_pose *= torch.tensor([ 1., 1.]).unsqueeze(0)


            targ = new_targ.clone()

            # targ = torch.zeros(new_targ.shape[0], new_targ.shape[1], 21)
            # targ[:, :, :7] = new_targ[:, :, 1:8]
            # targ[:, :, 7:14] = new_targ[:, :, 9:16]
            # targ[:, :, 14:21] = new_targ[:, :, 17:24]

            # confidence = torch.zeros(new_targ.shape[1], 3)
            # confidence[:, 0] = torch.sigmoid(new_targ[0, :, 0])
            # confidence[:, 1] = torch.sigmoid(new_targ[0, :, 8])
            # confidence[:, 2] = torch.sigmoid(new_targ[0, :, 16])

            patch_set = []
            for d_idx in tqdm(range(end-start)):
                t = targ[0, d_idx, :]
                k = 0
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                # ax should have xlim, ylim, grid at 0.2 step for x and y
                ax.set(xlim=(-1.0, 4.0), ylim=(-1, 1))
                ax.grid()

                patches = []
                rpos = np.array([rob_pos[d_idx, 0], rob_pos[d_idx, 1]])
                rrot = np.array(rob_pos[d_idx, 2]) * 180/np.pi
                rsize = np.array([0.588, 0.22])

                # print(new_pose.shape)

                # rpos_pred = np.array([new_pose[d_idx, 0], new_pose[d_idx, 1]])
                # rrot_pred = np.array(new_pose[d_idx, 2]) * 180/np.pi

                # robot patch
                patches.append(pch.Rectangle(rpos - rsize/2, *(rsize), angle=rrot, rotation_point='center', facecolor='green'))
                patches.append(pch.Rectangle(rpos - rsize/2, *(rsize), angle=rrot, rotation_point='center', facecolor='gray'))

                obs_pos = np.array(m_obstacle_data[d_idx, :2])
                if obs_pos[0] > 0:
                    # movable obstacle patch
                    obs_rot = np.array(m_obstacle_data[d_idx, 2])  * 180/np.pi
                    obs_size = np.array([0.43, 1.62])
                    patches.append(pch.Rectangle(obs_pos - obs_size/2, *(obs_size), angle=obs_rot, rotation_point='center', facecolor='black'))

                else:
                    patches.append(None)

                obs_pos = np.array(im_obstacle_data[d_idx, :2])
                if obs_pos[0] > 0:
                    # immovable obstacle patch
                    obs_rot = np.array(im_obstacle_data[d_idx, 2])  * 180/np.pi
                    obs_size = np.array([0.43, 1.42/2])
                    patches.append(pch.Rectangle(obs_pos - obs_size/2, *(obs_size), angle=obs_rot, rotation_point='center', facecolor='black'))
                else:
                    patches.append(None)
                    
                for i in range(2):
                    # contacts[i] = torch.sigmoid(t[k]).item()
                    color = 'red' if torch.sigmoid(t[k+1]) < 0.5 else 'yellow'
                    x, y, theta = t[k+2], t[k+3], t[k+4]
                    # if i == 1: theta = 0.
                    w, h = t[k+5], t[k+6]
                    k += 7
                    pos = np.array([x, y])
                    size = np.array([w, h])

                    if np.prod(size) < 0.1:
                        patches.append(None)
                        continue
                    angle = np.array([theta * 180 / np.pi])
                    # conf = confidence[d_idx, i].item()
                    patches.append(pch.Rectangle(pos - size/2, *(size), angle=angle, rotation_point='center', facecolor=color))
                patch_set.append(patches)
                plt.close()
            
            model_video_path = f'{HOME_DIR}/real_robot_videos/{folder}/{model_id}' 
            if not os.path.exists(model_video_path):
                os.makedirs(model_video_path)
            create_animation(patch_set, save_to=f'{model_video_path}', filename=f'video_{id}')

    

        
        


