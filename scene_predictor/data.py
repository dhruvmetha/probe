from torch.utils.data import Dataset
import numpy as np
from glob import glob
import torch

class TransformerDataset(Dataset):
    def __init__(self, files, sequence_length):
        self.all_folders = files
        self.sequence_length = int(sequence_length) + 1

    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        # if idx == 0:
        #     print(self.batch_sequence_segment[idx])
        # data = np.load(self.all_folders[idx])
        # inp, target, mask, fsw = torch.tensor(data['obs_hist']), torch.tensor(data['priv_obs']), torch.tensor(data['done']), torch.tensor(data['fsw'])
        data = np.load(self.all_folders[idx])

        # pose_inp = torch.cat([torch.zeros_like(torch.tensor(data['target'][0:1, :6])), torch.tensor(data['target'][:self.sequence_length-1, :6])], dim=0) # input with 0s concatenated in front (initial state).
        pose_target = torch.tensor(data['target'][1:self.sequence_length, :6]) # target is the difference between the current state and the previous state.
        
        # inp =  torch.cat([torch.tensor(data['input'][:self.sequence_length, :]), torch.tensor(data['actions'][:self.sequence_length, :])], dim=-1)
        inp =  torch.tensor(data['input'][1:self.sequence_length, :]) #torch.cat([, torch.tensor(data['actions'][:self.sequence_length, :])], dim=-1)
        
        target =  torch.cat([pose_target, torch.tensor(data['target'][1:self.sequence_length, 6:])] , dim=-1)
        
        mask, fsw = torch.tensor(data['done'][1:self.sequence_length]).unsqueeze(-1), torch.tensor(data['fsw'][1:self.sequence_length, :])

        return inp, target, mask, fsw


class TransformerDataset1(Dataset):
    def __init__(self, files, sequence_length, window_size=25):
        # self.folder = folder
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.window_size = int(window_size)
        self.batch_sequence_segment = np.zeros((len(self.all_folders)), dtype=np.int)

    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        # if idx == 0:
        #     print(self.batch_sequence_segment[idx])
        data = np.load(self.all_folders[idx])
        inp, target, mask, fsw = torch.tensor(data['obs_hist']), torch.tensor(data['priv_obs']), torch.tensor(data['done']), torch.tensor(data['fsw'])

        return inp, target, mask, fsw, idx

        inp_idx, targ_idx, mask_idx, fsw_idx = inp[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size], \
                            target[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size], \
                            mask[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size], \
                            fsw[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size]
        # if self.batch_sequence_segment[idx] == inp.shape[0] - self.window_size:
        if self.batch_sequence_segment[idx] == self.sequence_length - self.window_size:
            self.batch_sequence_segment[idx] = 0
        else:
            self.batch_sequence_segment[idx] += self.window_size
        return inp_idx, targ_idx, mask_idx, fsw_idx, self.batch_sequence_segment[idx] == self.window_size, idx

