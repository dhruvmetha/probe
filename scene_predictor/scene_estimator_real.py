from scene_estimator import SceneEstimator
from tqdm import tqdm
from glob import glob
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import pickle
import numpy as np
import torch
import os
from evaluate_iou import get_bbox_intersections_func

FFwriter = animation.FFMpegWriter

# HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-04_10-38-36'
HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-05_11-45-09'
# HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-06_07-47-52'
# HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-06_17-38-24'
HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-08_08-39-35'
HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-15_07-01-36'
HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-15_07-02-12'
# HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-16_16-27-10'
# HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-16_17-44-30'
HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/2023-09-21_21-02-08'
HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/2023-09-21_21-32-53'
HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/2023-09-21_22-53-00'
HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/2023-09-22_00-04-34'
HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/2023-09-22_00-22-04'
# HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/2023-09-22_01-53-44'

# /2023-09-22_00-04-34

DATA_DIR = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep1/2_obs/all_files'
RECTS = 3

def create_animation(patch_set, save_to=None, filename=None):

    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
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

    files_list = []
    folders_list = []

    # files = list(range(11)); folder = 'sep4'
    # files_list.append(files); folders_list.append(folder)
    # files = [1, 2, 3, 5]; folder = 'sep3'
    # files_list.append(files); folders_list.append(folder)
    # files = [8, 10, 13, 14, 15, 20]; folder = 'sep2'
    # files_list.append(files); folders_list.append(folder)

    # files = [3, 4, 5]; folder = 'sep14'
    # files_list.append(files); folders_list.append(folder)

    files = [2, 4, 5, 6]; folder = 'sep15'
    files_list.append(files); folders_list.append(folder)



    for files, folder in zip(files_list, folders_list):
        for file_no in files:
            # file_no = 8
            id = f'{file_no}'
            data_files = sorted(glob(f'/common/home/dm1487/Downloads/{folder}/{file_no}/*.npz'))[-1]

            data = np.load(data_files, allow_pickle=True)
            done_idx = data['done'].nonzero()[0][-1]
            start = 0
            end = done_idx
            inp = torch.from_numpy(data['input'][start:end]).unsqueeze(0)


            # setting up dof_pos, dof_vel, torque scaling
            inp[:, :, :3] *= 1
            inp[:, :, 3:15] *= 1
            inp[:, :, 15:27] *= 0.05

            # print(data['torques_estimated'].shape)
            inp[:, :, 27:39] = torch.from_numpy(data['torques_estimated'][start:end]).unsqueeze(0)
            inp[:, :, 27:39] *= 0.08

            # concatenating pose
            pose = (torch.from_numpy(data['target'][start:end]) * torch.tensor([0.25, 1, 1/3.14])).unsqueeze(0)
            # inp = torch.cat([inp, pose], dim=-1)
            inp = torch.cat([inp, pose], dim=-1)

            inp = torch.cat([inp, torch.zeros(1, 1500 - (end-start), inp.shape[2])], dim=1) # padding

            m_obstacle_data = data['obstacles'][0]
            im_obstacle_data = data['obstacles'][2]
            rob_pos = data['target']

            new_targ = se.predict(inp).to('cpu') * torch.tensor([1, 1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3).unsqueeze(0)

            targ = torch.zeros(new_targ.shape[0], new_targ.shape[1], 21)
            targ[:, :, :7] = new_targ[:, :, 1:8]
            targ[:, :, 7:14] = new_targ[:, :, 9:16]
            targ[:, :, 14:21] = new_targ[:, :, 17:24]

            confidence = torch.zeros(new_targ.shape[1], 3)
            confidence[:, 0] = torch.sigmoid(new_targ[0, :, 0])
            confidence[:, 1] = torch.sigmoid(new_targ[0, :, 8])
            confidence[:, 2] = torch.sigmoid(new_targ[0, :, 16])

            patch_set = []
            ground_truth = []
            pred = []
            
            for d_idx in tqdm(range(end-start)):
                t = targ[0, d_idx, :]
                k = 0
                fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
                # ax should have xlim, ylim, grid at 0.2 step for x and y
                ax.set(xlim=(-1.0, 4.0), ylim=(-1, 1))
                ax.axis('off')

                patches = []
                rpos = np.array([rob_pos[d_idx, 0], rob_pos[d_idx, 1]])
                rrot = np.array(rob_pos[d_idx, 2]) * 180/np.pi
                rsize = np.array([0.52, 0.30])

                # robot patch
                patches.append(pch.Rectangle(rpos - rsize/2, *(rsize), angle=rrot, rotation_point='center', facecolor='green'))
                gt_boxes = []                
                obs_pos = np.array(m_obstacle_data[d_idx, :2])
                gt_boxes.extend([0., 0., 0., 0., 0.])
                if obs_pos[0] > 0:
                    # movable obstacle patch
                    obs_rot = np.array(m_obstacle_data[d_idx, 2])  * 180/np.pi
                    obs_size = np.array([0.43, 1.62])
                    patches.append(pch.Rectangle(obs_pos - obs_size/2, *(obs_size), angle=obs_rot, rotation_point='center', facecolor='black'))
                    gt_boxes[:5] = [obs_pos[0], obs_pos[1], obs_rot*np.pi/180, obs_size[0], obs_size[1]]

                else:
                    patches.append(None)

                obs_pos = np.array(im_obstacle_data[d_idx, :2])
                gt_boxes.extend([0., 0., 0., 0., 0.])
                if obs_pos[0] > 0:
                    # immovable obstacle patch
                    obs_rot = np.array(im_obstacle_data[d_idx, 2])  * 180/np.pi
                    obs_size = np.array([0.43, 0.62])
                    patches.append(pch.Rectangle(obs_pos - obs_size/2, *(obs_size), angle=obs_rot, rotation_point='center', facecolor='black'))
                    gt_boxes[5:] = [obs_pos[0], obs_pos[1], obs_rot*np.pi/180, obs_size[0], obs_size[1]]
                else:
                    patches.append(None)
                
                pred_boxes = [0.] * 10
                for i in range(2):
                    conf = 1.0 if (confidence[d_idx, i].item() > 0.8) else 0.0
                    # contacts[i] = torch.sigmoid(t[k]).item()
                    # print(conf)
                    color = 'red' if torch.sigmoid(t[k+1]) < 0.5 else 'yellow'
                    x, y, theta = t[k+2], t[k+3], t[k+4]
                    # if i == 1: theta = 0.
                    w, h = t[k+5], t[k+6]
                    pred_boxes[i*5:i*5+5] = [x, y, theta, w, h]
                    k += 7
                    pos = np.array([x, y])
                    size = np.array([w, h])

                    if np.prod(size) < 0.12:
                        patches.append(None)
                        continue
                    angle = np.array([theta * 180 / np.pi])
                    # conf = confidence[d_idx, i].item()
                    patches.append(pch.Rectangle(pos - size/2, *(size), angle=angle, rotation_point='center', facecolor=color, alpha=conf))
                patch_set.append(patches)
                plt.close()

                ground_truth.append(gt_boxes)
                pred.append(pred_boxes)

            ground_truth = np.array(ground_truth)
            ground_truth = ground_truth[np.newaxis, :, :]

            pred = np.array(pred)
            pred = pred[np.newaxis, :, :]

            mask = np.ones((1, ground_truth.shape[1], 1500))
            mask[:, :, end-start:] = 0

            evaluation_path = f'{HOME_DIR}/real_robot_evaluations/{folder}/{model_id}'
            evaluation_data = {
                'gt': ground_truth,
                'pred': pred,
                'mask': mask
            }

            # save as pickle
            if not os.path.exists(evaluation_path):
                os.makedirs(evaluation_path)
            with open(f'{evaluation_path}/evaluation_{id}.pkl', 'wb') as f:
                pickle.dump(evaluation_data, f)
            result = get_bbox_intersections_func([evaluation_data])

            # save result in the real_robot_evaluations folder as pickle
            with open(f'{evaluation_path}/result_{id}.pkl', 'wb') as f:
                pickle.dump(result, f)


            # for k, v in result.items():
            #     for k1, v1 in v.items():
            #         if k == 'fixed':
            #             print(k1, v1)
            #         print(f'{k}_{k1}: {np.mean(v1)}')




            plt.close()
            import pandas as pd

            
                # import matplotlib.pyplot as plt

                # Assuming result['movable']['iou'] and result['fixed']['iou'] are your IOU data

                # Define the window size for the moving average
            window_size = 5 

            kernel_size = 10
            kernel = np.ones(kernel_size) / kernel_size
            # data_convolved = np.convolve(data, kernel, mode='same')

            # print(result['movable']['iou'])
            # Calculate the moving average for both 'movable' and 'fixed'
            try:
                movable_iou_smoothed = np.convolve(np.array(result['movable']['iou']), kernel, mode='same') # .rolling(window=window_size).mean()
            except:
                movable_iou_smoothed = np.array(result['movable']['iou'])
            
            try:
                fixed_iou_smoothed = np.convolve(result['fixed']['iou'], kernel, mode='same')
            except:
                fixed_iou_smoothed = np.array(result['fixed']['iou'])

                # Create a figure and plot the smoothed data
            plt.figure(figsize=(25.0, 10.80), dpi=100)
            plt.plot(movable_iou_smoothed, 'r', label='movable (smoothed)')
            plt.plot(fixed_iou_smoothed, 'b', label='fixed (smoothed)')

            # Customize the plot
            # plt.rc('xtick', labelsize=32)
            # plt.rc('ytick', labelsize=32)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.title('Smoothed IOU Chart for the Real Robot Prediction', fontsize=24)
            plt.xlabel("Time", fontsize=24)
            plt.ylabel("IOU", fontsize=24)
            plt.grid()
            plt.legend(fontsize=24)

            # Show the plot or save it as an image
            plt.show()

            # Optionally, save the plot as an image
            plt.savefig(f'{evaluation_path}/iou_{id}_smoothed.png', dpi=300)
            

            # plt.figure(figsize=(10, 10))
            # plt.rc('xtick',labelsize=16)
            # plt.rc('ytick',labelsize=16)
            # plt.title('IOU Chart for the Real Robot Prediction',fontsize=16)
            # plt.xlabel("Time",fontsize=16)
            # plt.ylabel("IOU",fontsize=16)
            # plt.plot(result['movable']['iou'], 'r', label='movable')
            # plt.plot(result['fixed']['iou'], 'b', label='fixed')
            # plt.grid()
            # plt.show()
            # plt.legend()
            # plt.savefig(f'{evaluation_path}/iou_{id}.png', dpi=300)
            
            model_video_path = f'{HOME_DIR}/real_robot_videos/{folder}/{model_id}' 
            if not os.path.exists(model_video_path):
                os.makedirs(model_video_path)
            create_animation(patch_set, save_to=f'{model_video_path}', filename=f'video_{id}')

            

    

        
        


