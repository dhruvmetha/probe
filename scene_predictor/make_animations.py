import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import glob
import os
from tqdm import tqdm 
from pathlib import Path
import time
from indep_model.config import *

FFwriter = animation.FFMpegWriter
estimate_pose = False
# RECENT_MODEL = sorted(glob.glob(f'./scene_predictor/results_{"results_1_obs_pe" if estimate_pose else "priv_info"}/*/*'), key=os.path.getmtime)[-1]

# RECENT_MODEL = sorted(glob.glob(f'./scene_predictor/results/transformer/full_from_qqdottau/*'))[-1]
RECENT_MODEL = sorted(glob.glob(f'./scene_predictor/results/transformer/2_obs/qqdtaupose_to_cdctmvposesize/*'))[-1]
# RECENT_MODEL = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results/transformer/2_obs/full_prediction/2024-02-22_02-45-10'
# RECENT_MODEL = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results/transformer_750_2048/2023-05-20_00-36-45'
# RECENT_MODEL = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results/transformer_750_2048/2023-05-20_15-21-17'
# RECENT_MODEL = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-08-28_18-51-51'
print(RECENT_MODEL)

source_folder = f"{RECENT_MODEL}/viz"
dest_folder = f"{RECENT_MODEL}/animations"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
RECTS = 2

def on_new_file(tmp_img_path, save_to=None):

    file_name = Path(tmp_img_path).stem
    # if os.path.exists(f"{dest_folder}/{file_name}.mp4"):
    #     continue
    fig, axes = plt.subplots(2, 2, figsize=(48, 24))
    ax = axes.flatten()

    try:
        with open(tmp_img_path, 'rb') as f:
            patches = pickle.load(f)
    except:
        plt.close()
        return False
    
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
        
        if not estimate_pose:
            for i in range(RECTS):
                j = i*6 + 5
                ax[0].add_patch(frame[j])
                ax[0].add_patch(frame[j+1])

                ax[1].add_patch(frame[j+3])
                ax[2].add_patch(frame[j+4])

                ax[3].add_patch(frame[j+5])

        last_patch.extend(frame)
    

    num_patches = min(1000, len(patches))
    anim = animation.FuncAnimation(fig, animate, frames=patches[:num_patches], interval=10, repeat=False)
    if save_to is None:
        anim.save(f"{dest_folder}/{file_name}.mp4", writer = FFwriter(20))
    else:
        anim.save(f"{save_to}/{file_name}.mp4", writer = FFwriter(20))
    plt.close()
    return True


if __name__ == '__main__':

    file_list = []

    print(source_folder)
    while True:
        # get the updated list of files in the folder
        if not os.path.exists(source_folder):
            time.sleep(5)
            continue
        updated_file_list = sorted(glob.glob(f"{source_folder}/*.pkl"), key= lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))[-3:][::-1]
        print(updated_file_list)

        # check for new files
        for file_name in list(updated_file_list):
            if file_name not in file_list:
                done = False
                # call the function when a new file is detected
                while not done:
                    print('working on', Path(file_name).stem)
                    done = on_new_file(file_name)
                    print('done', Path(file_name).stem)

        # update the list of files
        file_list = updated_file_list

        # sleep for a bit before checking again
        time.sleep(1)