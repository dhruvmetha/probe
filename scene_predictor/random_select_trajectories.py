import glob
import random
import os


files = glob.glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/2_obs/3/*/*.npz')
random.shuffle(files)

# create new folder in legged_manipulation_data_store called visualization_data
folder = '/common/users/dm1487/legged_manipulation_data_store/visualization_data'
if not os.path.exists(folder):
    os.makedirs(folder)

# copy 5 files to visualization_data
for i in range(5):
    os.system('cp ' + files[i] + ' ' + folder)

