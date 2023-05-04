from pathlib import Path
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import random

import pickle

if __name__ == '__main__':
    dest_path = Path(f'/home/dhruv/projects_dhruv/rollout_data_single_trajectories')
    files = glob(str(dest_path/'*/*.npz'))
    file = random.choice(files)
    
    data = np.load(file)


    pos = data['target'][:-1, :2]
    plt.xlim(-0.5, 3.5)   
    plt.ylim(-1.0, 1.0)
    plt.plot(pos[:, 0], pos[:, 1])
    plt.show()
        
        
    
    
        
