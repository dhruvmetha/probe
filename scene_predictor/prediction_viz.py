from visualization import get_visualization, get_animation
# from make_animations import make_animation
import numpy as np
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import torch

SAVE_FILE_NAME = 'prediction_1_single_trajectories'
data_path = f"/common/users/dm1487/legged_manipulation_data/rollout_data/{SAVE_FILE_NAME}/*/*.npz"

all_files = glob.glob(data_path)
print(all_files)
for file in all_files:
    data = np.load(file)
    prediction = data['prediction']
    target = data['target']
    mask = data['done']
    fsw_data = data['fsw']
    patches = []
    print(target.shape)
    for step in range(target.shape[0]):

        pred_obs = torch.tensor(prediction[step, :6]).unsqueeze(0)
        pred = torch.tensor(prediction[step, 6:]).unsqueeze(0)

        obs = torch.tensor(target[step, :6]).unsqueeze(0)
        targ = torch.tensor(target[step, 6:]).unsqueeze(0)
        fsw = torch.tensor(fsw_data[step, :]).unsqueeze(0)

        # print(obs.shape)

        patch_set = get_visualization(0, obs, targ, pred_obs, pred, fsw)
        patches.append(patch_set)
    anim = get_animation(patches)
    plt.show()

    