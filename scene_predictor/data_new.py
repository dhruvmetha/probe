from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch

class TransformerDataset(Dataset):
    def __init__(self, cfg, files, sequence_length, estimate_pose=False):
        
        self.output_size = sum(cfg.outputs.values())
        self.obstacles = cfg.obstacles
        self.input_dict = cfg.inputs
        self.output_dict = cfg.outputs

        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.estimate_pose = estimate_pose
    
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])
        
        # data keys: input, target, target_env, actions, ll_actions, fsw, done 

        # input data
        inp = torch.tensor(data['input']) # project_gravity 3, joint_pos 12, joint_vel 12, torques 12

        # split inp
        projected_gravity = inp[:, :3]
        joint_pos = inp[:, 3:15]
        joint_vel = inp[:, 15:27]
        torques = inp[:, 27:39] * 0.08 # scale torques
        pose = torch.tensor(data['target'])[:, :3]
        inputs = {
            'projected_gravity': projected_gravity,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'torques': torques, # scale torques,
            'pose': pose
        }

        inputs = [inputs[k] for k in self.input_dict.keys()]
        # print(inputs)
        # print("#########################")
        inputs = torch.cat(inputs, dim=-1)

        done_idx = data['done'][:].nonzero()[0][-1]
        
        # target data
        target = torch.tensor(data['target_env'])
        fsw = torch.tensor(data['fsw'])
        final_target = torch.zeros((self.sequence_length, int(self.obstacles * self.output_size)))
        final_fsw = torch.zeros((self.sequence_length, int(self.obstacles * self.output_size)))
        
        k = 0
        for obs_idx in range(self.obstacles):
            true_k = (9 * obs_idx) # skip the last 2 elements (data contains: mass and friction that we do not consider in the model)
            if 'contact' in self.output_dict:
                final_target[:, k] = target[:, true_k + 0]
                final_fsw[:, k] = fsw[:, true_k + 0]
                k += self.output_dict['contact']
            if 'movable' in self.output_dict:
                final_target[:, k] = target[:, true_k + 1]
                final_fsw[:, k] = fsw[:, true_k + 1]
                k += self.output_dict['movable']
            if 'pose' in self.output_dict:
                final_target[:, k:k+self.output_dict['pose']] = target[:, true_k + 2: true_k + 5] * torch.tensor([0.25, 1, 1/3.14])
                final_fsw[:, k:k+self.output_dict['pose']] = fsw[:, true_k + 2: true_k + 5]
                k += self.output_dict['pose']
            if 'size' in self.output_dict:
                final_target[:, k:k+self.output_dict['size']] = target[:, true_k + 5: true_k + 7] * torch.tensor([1, 1/1.7])
                final_fsw[:, k:k+self.output_dict['size']] = fsw[:, true_k + 5: true_k + 7]
                k += self.output_dict['size']
        # final_target[:, :7] = target[:, :7]
        # final_target[:, 7:] = target[:, 9:16]
        # final_target *= torch.tensor([1, 1, 0.25, 1, 1/3.14, 1, 1/1.7] * 2)

        # teacher forcing
        shifted_final_target = torch.zeros((self.sequence_length, int(self.obstacles * self.output_size)))
        shifted_final_target[1:, :] = final_target.clone()[:-1, :]
        teacher_force = False
        if teacher_force:
            inputs = torch.cat([inputs, shifted_final_target], dim=-1)
        
        # final_fsw = torch.zeros((self.sequence_length, int(self.obstacles * self.output_size)))
        # fsw = torch.tensor(data['fsw'])
        # final_fsw[:, :7] = fsw[:, :7]
        # final_fsw[:, 7:] = fsw[:, 9:16]

        mask = torch.tensor(data['done'])[:].unsqueeze(-1)

        # print(fsw.shape)
        return inputs, final_target, mask, final_fsw, pose, target
    
if __name__ == '__main__':

    import pickle
    with open('/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24/balanced/train_1.pkl', 'rb') as f:
        files = pickle.load(f)
    files = files[:10]
    dataset = TransformerDataset(files, 1500)
    dataset[0]
    