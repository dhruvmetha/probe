from torch.utils.data import Dataset
import numpy as np
from glob import glob
import torch

class TransformerDataset(Dataset):
    def __init__(self, files, sequence_length):
        self.all_folders = files
        self.sequence_length = int(sequence_length)

    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])
        
        torques = torch.tensor(data['torques'][1:self.sequence_length, :])
        actions = torch.clamp(torch.tensor(data['actions'][1:self.sequence_length, :]), -0.65, 0.65)
        inp_obs = torch.tensor(data['input'][:self.sequence_length-1, :])
        projected_gravity = inp_obs[:self.sequence_length-1, :3]
        joint_pos = inp_obs[:self.sequence_length-1, 18:30]
        joint_vel = inp_obs[:self.sequence_length-1, 30:42]

        inp = torch.cat([projected_gravity, joint_pos, joint_vel, torques, actions], dim=-1) # 3 + 12 + 12 + 12 + 3 = 42
        target = torch.tensor(data['target'][1:self.sequence_length, :])

        mask, fsw = torch.tensor(data['done'][:self.sequence_length-1]).unsqueeze(-1), torch.tensor(data['fsw'][:self.sequence_length-1, :])

        return inp, target, mask, fsw