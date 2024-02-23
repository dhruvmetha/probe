from matplotlib import pyplot as plt
from matplotlib import patches as pch
import numpy as np
import pickle
import random
import time
from PIL import Image

# main if __name__ == "__main__" block
if __name__ == "__main__":
    
    file_list = '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb22/2_obs/balanced/train_2.pkl'
    
    with open(file_list, 'rb') as f:
        all_files = pickle.load(f)
    
    random.shuffle(all_files)
    file = all_files[0]
    data = np.load(file)

    fig, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=100)
    
    done_idx = data['done'].nonzero()[0][-1]
    outs = np.zeros((done_idx, 150, 300, 3))
    obs1 = data['target_env'][:, 2:7]
    obs2 = data['target_env'][:, 11:16]
    
    st = time.time()
    for i in range(done_idx):
        plt.cla()

        x, y, angle, size_x, size_y = obs1[i]
        angle = np.rad2deg(angle)
        rect = pch.Rectangle(np.array([x, y]) - np.array([size_x, size_y])/2, size_x, size_y, angle=angle, rotation_point='center', facecolor='blue')
        ax.add_patch(rect)
        
        x, y, angle, size_x, size_y = obs2[i]
        angle = np.rad2deg(angle)
        rect = pch.Rectangle(np.array([x, y]) - np.array([size_x, size_y])/2, size_x, size_y, angle=angle, rotation_point='center', facecolor='red')
        ax.add_patch(rect)

        # set the axis limits
        ax.set(xlim=(0.0, 4.0), ylim=(-1, 1))
        ax.axis('off')

        plt.savefig('./tmp.png', format='png', )
        img = Image.open('./tmp.png')
        image_array = np.array(img)
        outs[i] = image_array[:, :, :3]

        
    new_data = dict(data)
    new_data['obs_img'] = outs
    np.savez_compressed('./tmp.npz', **new_data)
    print(time.time() - st)
            