from matplotlib import pyplot as plt
import pickle
import random
import numpy as np
from tqdm import tqdm

data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_aug27/2_obs/all_files/all_files_3.pkl'

with open(data_folder, 'rb') as f:
    balanced_data = pickle.load(f)
# balanced_data = glob(data_folder)
random.shuffle(balanced_data)
print('# trajectories:', len(balanced_data))
num_envs = 1000
files = random.sample(balanced_data, num_envs)

joints_min = {}
joints_max = {}

for idx, file in tqdm(enumerate(files)):
    data = np.load(file)
    done_idx = data['done'].nonzero()[0][-1]
    for i in range(12):
        if i not in joints_min.keys():
            joints_min[i] = []
            joints_max[i] = []
        joints_min[i].append(np.min(data['input'][:done_idx, 27+i]/12))
        joints_max[i].append(np.max(data['input'][:done_idx, 27+i]/12))
    
# find and print mean min and max for each joint
for i in [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]:
    print('######## JOINT', i, '########')
    print(i, np.mean(joints_min[i]), np.mean(joints_max[i]))
    print(i, np.median(joints_min[i]), np.median(joints_max[i])) # almost same as mean
    print(i, np.std(joints_min[i]), np.std(joints_max[i]))
    print(i, np.min(joints_min[i]), np.min(joints_max[i]))
    print(i, np.max(joints_min[i]), np.max(joints_max[i]))
    print()

# 1/12 = 0.08333333333333333

# ######## JOINT 0 ########
# 0 -9.225248 3.2856693 range = 12.510917
# 0 1.0276737 1.0650898
# 0 -15.402886 1.5742388
# 0 -6.4038796 8.811499

# ######## JOINT 3 ########
# 3 -3.127337 8.153692 range = 11.281029
# 3 1.0354599 0.904139
# 3 -8.627178 5.74954
# 3 -1.4765756 11.022001

# ######## JOINT 6 ########
# 6 -8.604232 3.4757202 range = 12.079952
# 6 1.3516488 0.8672834
# 6 -13.767919 1.5022472
# 6 -5.3952684 8.301601

# ######## JOINT 9 ########
# 9 -3.4816868 8.752332 range = 12.234019
# 9 1.2452147 1.0803834
# 9 -10.280777 5.67347
# 9 -1.5302817 12.503623

# ######## JOINT 1 ########
# 1 -5.151589 5.3233576 range = 10.474946
# 1 1.835629 1.60563
# 1 -12.124653 1.843465
# 1 -1.8523715 9.896796

# ######## JOINT 4 ########
# 4 -4.8125067 6.4997044 range = 11.312211
# 4 1.8343892 1.1951044
# 4 -13.469277 2.6667035
# 4 -2.0556931 11.235469

# ######## JOINT 7 ########
# 7 -5.645478 6.2180476     range = 11.863525
# 7 1.4929638 1.6086453
# 7 -11.074146 2.3788633
# 7 -2.776413 13.492264


# ######## JOINT 10 ########
# 10 -4.567364 8.578365 range = 13.145729
# 10 1.1575108 2.4463694
# 10 -9.159012 2.5284564
# 10 -2.1306102 16.34842


# ######## JOINT 2 ########
# 2 -2.2737184 16.25696 range = 18.530678
# 2 1.7149129 1.1532328
# 2 -11.597355 13.247574
# 2 -0.8337027 19.873531

# ######## JOINT 5 ########
# 5 -1.8842659 15.769403 range = 17.653669
# 5 1.166861 1.5177491
# 5 -9.856052 12.286763
# 5 -0.7820343 20.419712


# ######## JOINT 8 ########
# 8 -1.8492664 18.134956 range = 19.984222
# 8 0.7966499 1.741537 
# 8 -6.6685395 13.448775
# 8 -0.56913435 23.156319


# ######## JOINT 11 ########
# 11 -2.0150354 18.77798 range = 20.793015
# 11 0.66083854 1.7392586
# 11 -7.899699 14.299902
# 11 -1.0097648 24.067774
