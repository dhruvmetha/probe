from matplotlib import pyplot as plt
import pickle
import torch
import os
import glob
from pathlib import Path
from ml_logger import logger
from go1_gym import MINI_GYM_ROOT_DIR
import numpy as np

recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/high_level_policy/*/*/*"), key=os.path.getmtime)
model_path = recent_runs[-1]
logger.configure(Path(model_path).resolve())

# read pickle file
# with open('eval_torques/torques_632.pkl', 'rb') as f:
#     data = pickle.load(f)

data = logger.load_pkl('eval_torques/torques_1130.pkl')
# print(torch.square(data[0][0]))
# print(data[0][0].shape)
# all_torques = torch.tensor(data[0])
# find the squared sum of the torch vector data
# and plot it
squared_sum = []
for i in range(len(data[0])):
    squared_sum.append(torch.sum(torch.square(data[0][i])).cpu().numpy())
    # for j in range(12):
    #     if j not in squared_sum:
    #         squared_sum[j] = []
    #     squared_sum[j].append(data[0][i][j].cpu().numpy())

# for k, v in squared_sum.items():
#     smoothed = np.convolve(np.array(v), np.ones(20)/20)
#     plt.plot(smoothed, label=f"joint {k}")

# plt.legend()

# squared_sum = np.convolve(squared_sum, np.ones(10)/10)
# print(squared_sum)
plt.plot(squared_sum)
plt.show()