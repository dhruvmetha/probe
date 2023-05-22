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
        

        # input data
        torques = torch.tensor(data['torques'][1:self.sequence_length, :])
        actions = torch.clamp(torch.tensor(data['actions'][1:self.sequence_length, :]), -0.65, 0.65)
        inp_obs = torch.tensor(data['input'][1:self.sequence_length, :])
        projected_gravity = inp_obs[:, :3]
        joint_pos = inp_obs[:, 18:30]
        joint_vel = inp_obs[:, 30:42]

        pose = torch.tensor(data['target'][1:self.sequence_length, :6])
        priv_info = torch.tensor(data['target'][1:self.sequence_length, 6:])

        # print(projected_gravity.shape, joint_pos.shape, joint_vel.shape, torques.shape, pose.shape, actions.shape)

        # inp = torch.cat([projected_gravity, joint_pos, joint_vel, torques, pose[1:self.sequence_length-1, :], actions], dim=-1) # 3 + 12 + 12 + 12 + 6 + 3 = 48
        inp = torch.cat([torques, projected_gravity, joint_pos, joint_vel, pose], dim=-1) # 3 + 12 + 12 + 12 + 6 + 3 = 48
        
        # target data
        
        # target_joint_pos = torch.tensor(data['input'][2:self.sequence_length, 18:30])
        # target_joint_vel = torch.tensor(data['input'][2:self.sequence_length, 30:42])
        target_pose = torch.tensor(data['target'][2:self.sequence_length, :6])
        target_priv_info = torch.tensor(data['target'][2:self.sequence_length, 6:])
        # target = torch.cat([target_pose, target_priv_info], dim=-1)
        target = torch.cat([priv_info], dim=-1)

        # delta prediction target
        # target = torch.cat([target_pose - pose[:self.sequence_length-1]], dim=-1)

        mask, fsw = torch.tensor(data['done'][1:self.sequence_length]).unsqueeze(-1), torch.tensor(data['fsw'][1:self.sequence_length, :])

        # print(inp[0, -9:-6], target[0, -27:-24])
        # print(inp[1, -9:-6], target[1, -27:-24])
        # print(inp[2, -9:-6], target[2, -27:-24])

        return inp, target, mask, fsw, pose