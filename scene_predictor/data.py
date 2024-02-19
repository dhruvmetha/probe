from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch

class TransformerDataset(Dataset):
    def __init__(self, files, sequence_length, estimate_pose=True):
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.estimate_pose = estimate_pose
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])
        
        # keys: input, target, target_env, actions, ll_actions, fsw, done 
        
        # input data
        torques = torch.tensor(data['torques'][1:self.sequence_length, :])
        actions = torch.clamp(torch.tensor(data['actions'][1:self.sequence_length, :]), -0.4, 0.4)
        inp_obs = torch.tensor(data['input'][1:self.sequence_length, :])
        projected_gravity = inp_obs[:, :3]
        joint_pos = inp_obs[:, 18:30]
        joint_vel = inp_obs[:, 30:42]

        pose = torch.tensor(data['target'][1:self.sequence_length, :6])
        priv_info = torch.tensor(data['target'][1:self.sequence_length, 6:])

        # print(projected_gravity.shape, joint_pos.shape, joint_vel.shape, torques.shape, pose.shape, actions.shape)

        # inp = torch.cat([projected_gravity, joint_pos, joint_vel, torques, pose[1:self.sequence_length-1, :], actions], dim=-1) # 3 + 12 + 12 + 12 + 6 + 3 = 48
        
        # if self.estimate_pose:
        #     pose_diff = pose[1:self.sequence_length-1, :2] - pose[:self.sequence_length-2, :2]
        #     inp = torch.cat([projected_gravity, joint_pos, joint_vel, pose], dim=-1) # 3 + 12 + 12 + 12 + 6 + 3 = 48
        #     target = torch.cat([pose_diff, pose[:, 2]], dim=-1)
        #     print(target.shape)
        # else:

            # inp = torch.cat([torques, projected_gravity, joint_pos, joint_vel, pose], dim=-1) # 3 + 12 + 12 + 12 + 6 + 3 = 48
        inp = torch.cat([torques, projected_gravity, joint_pos, joint_vel, pose], dim=-1) # 3 +_12 + 12 + 6 = 33
        
        # target data
        
        # target_joint_pos = torch.tensor(data['input'][2:self.sequence_length, 18:30])
        # target_joint_vel = torch.tensor(data['input'][2:self.sequence_length, 30:42])
        # target_pose = torch.tensor(data['target'][2:self.sequence_length, :6])
        # target_priv_info = torch.tensor(data['target'][2:self.sequence_length, 6:])
        # target = torch.cat([target_pose, target_priv_info], dim=-1)
        target = torch.cat([priv_info * torch.tensor([1, 1, 0.33, 1, 1/3.14, 1, 1/1.7] * 3)], dim=-1)

        # delta prediction target
        # target = torch.cat([target_pose - pose[:self.sequence_length-1]], dim=-1)

        mask, fsw = torch.tensor(data['done'][1:self.sequence_length]).unsqueeze(-1), torch.tensor(data['fsw'][1:self.sequence_length, :])

        # print(inp[0, -9:-6], target[0, -27:-24])
        # print(inp[1, -9:-6], target[1, -27:-24])
        # print(inp[2, -9:-6], target[2, -27:-24])

        # print(inp[0], inp[1], inp[2])
        # print(target[0], target[1], target[2])

        return inp, target, mask, fsw, pose
    


class PoseEstimatorDataset(Dataset):
    def __init__(self, files, sequence_length, estimate_pose=True):
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.estimate_pose = estimate_pose
    
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])

        # keys: input, target, target_env, actions, ll_actions, fsw, done 
        
        # noise
        # dof_pos = 0.01
        # dof_vel = 1.5
        noise_scale_vec = torch.tensor([0.01] * 12 + [1.5] * 12)
        inp = torch.tensor(data['input'])[:, :24]  
        inp += (2 * torch.rand_like(inp) - 1) * noise_scale_vec

        target = torch.tensor(data['target'])[:, :5]
        
        target_diff = torch.cat([torch.zeros(1, 5), target[1:, :] - target[:-1, :]], dim=0)
        target_diff[:, :3] *= 10
        target_diff[:, 3:] *= 1

        target *= torch.tensor([0.25, 1, 1/3.14, 1/0.4, 1/0.4])
        mask, fsw = torch.tensor(data['done']).unsqueeze(-1), torch.tensor(data['fsw'])

        return inp, target, mask, fsw, target_diff
    
class PoseDiffEstimatorDataset(Dataset):
    def __init__(self, files, sequence_length, estimate_pose=True):
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.estimate_pose = estimate_pose
    
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])

        # keys: input, target, target_env, actions, ll_actions, fsw, done 
        
        # noise
        # dof_pos = 0.01
        # dof_vel = 1.5
        noise_scale_vec = torch.tensor([0.01] * 12 + [1.5] * 12)
        inp = torch.tensor(data['input'])[:, :24]  
        inp += (2 * torch.rand_like(inp) - 1) * noise_scale_vec

        target = torch.cat([torch.zeros(1, 5), torch.tensor(data['target'])[:-1, :5]], dim=0)
        target_diff = torch.cat([torch.zeros(1, 5), target[1:, :] - target[:-1, :]], dim=0)
        target_diff[:, :3] *= 100
        target_diff[:, 3:] *= 1
        target *= torch.tensor([0.25, 1, 1/3.14, 1/0.4, 1/0.4])
        inp_w_target = torch.cat([inp, target], dim=-1)
        target /= torch.tensor([0.25, 1, 1/3.14, 1/0.4, 1/0.4])
        target[:, :3] *= 100

        mask, fsw = torch.tensor(data['done']).unsqueeze(-1), torch.tensor(data['fsw'])

        return inp_w_target, target, mask, fsw, target_diff
    
class SceneEstimatorDataset(Dataset):
    def __init__(self, files, sequence_length, start=10, estimate_pose=True):
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.estimate_pose = estimate_pose
        self.start = start
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])
        start = self.start
        # keys: input, target, target_env, actions, ll_actions, fsw, done 
        inp = torch.tensor(data['input'])
        inp[:, 27:39] *= 0.08
        # print(inp.shape)
        
        pose = torch.tensor(data['target'])[:, :3] # * torch.tensor([0.25, 1, 1/3.14])
        done_idx = data['done'][:].nonzero()[0][-1]
        target = data['target_env'][:, :]

        # print(pose[:2])
        # enable code below for noisy pose 
        # randomly induce a few repeated poses to make the model learn to ignore the pose
        
        # repeat_count = int(np.random.uniform(0.05, 0.15) * done_idx)
        
        # if np.random.uniform() < 0.75:
        #     if np.random.uniform() < 0.75:
        #         random_indices = torch.randint(2, done_idx, (repeat_count,))
        #         pose[random_indices, :] = pose[random_indices-2, :]
        #         pose[random_indices-1, :] = pose[random_indices-2, :]
        #     else:
        #         random_indices = torch.randint(3, done_idx, (repeat_count,))
        #         pose[random_indices, :] = pose[random_indices-3, :]
        #         pose[random_indices-1, :] = pose[random_indices-3, :]
        #         pose[random_indices-2, :] = pose[random_indices-3, :]


        
        confidence_target = np.zeros((target.shape[0], 24))
        # confidence_target = np.concatenate([np.zeros((target.shape[0], 1)), target], axis=1)

        confidence_target[:, 1:8] = target[:, :7]
        confidence_target[:, 9:16] = target[:, 7:14]
        confidence_target[:, 17:24] = target[:, 14:21]
        
        if len(target[:, 0].nonzero()[-1]) > 0:
            contact_idx = target[:, 0].nonzero()[-1][0]
            confidence_target[contact_idx:done_idx, 0] = 1.
        
        if len(target[:, 7].nonzero()[-1]) > 0:
            contact_idx = target[:, 7].nonzero()[-1][0]
            confidence_target[contact_idx:done_idx, 8] = 1.

        if len(target[:, 14].nonzero()[-1]) > 0:
            contact_idx = target[:, 14].nonzero()[-1][0]
            confidence_target[contact_idx:done_idx, 16] = 1.

        target = torch.tensor(confidence_target).to(torch.float32)

        target *= torch.tensor([1, 1, 1, 0.25, 1, 1/3.14, 1, 1/1.7] * 3)
        
        # target = data['target_env']
        # new_target = np.zeros((target.shape[0], 24))
        # new_target[:, :4] = target[:, :4]
        # new_target[:, 4] = np.sin(target[:, 4])
        # new_target[:, 5] = np.cos(target[:, 4])
        # new_target[:, 6:8] = target[:, 5:7]
        
        # new_target[:, 8:12] = target[:, 7:11]
        # new_target[:, 12] = np.sin(target[:, 11])
        # new_target[:, 13] = np.cos(target[:, 11])
        # new_target[:, 14:16] = target[:, 12:14]

        # new_target[:, 16:20] = target[:, 14:18]
        # new_target[:, 20] = np.sin(target[:, 18])
        # new_target[:, 21] = np.cos(target[:, 18])
        # new_target[:, 22:24] = target[:, 19:21]

        # target = torch.tensor(new_target).to(torch.float32)
        # target *= torch.tensor([1, 1, 0.25, 1, 1., 1., 1., 1/1.7] * 3)

        mask, fsw = torch.tensor(data['done'])[:].unsqueeze(-1), torch.tensor(data['fsw'])[:]
        
        return torch.cat([inp, pose], dim=-1), target, mask, fsw, pose

class PoseEstimatorGRUDataset(Dataset):
    def __init__(self, files, sequence_length):
        self.all_folders = files
        self.sequence_length = int(sequence_length)
    
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])
        
        # keys: input, target, target_env, actions, ll_actions, fsw, done 

        inp = torch.tensor(data['input'])[:, :24]
        target = torch.tensor(data['target'])[:, :5]
        target_diff = torch.cat([torch.zeros(1, 5), target[1:, :] - target[:-1, :]], dim=0)
        target_diff[:, :3] *= 10
        target_diff[:, 3:] *= 1
        # print(target_diff[:, :3] * 100, target_diff[:, 3:] * 10)
        # print(target_diff.shape)
        target *= torch.tensor([0.25, 1, 1/3.14, 1/0.4, 1/0.4])

        
        
        # print(target.shape, target_diff.shape, target_diff[0])
        mask, fsw = torch.tensor(data['done']).unsqueeze(-1), torch.tensor(data['fsw'])

        # print(inp[0, -9:-6], target[0, -27:-24])
        # print(inp[1, -9:-6], target[1, -27:-24])
        # print(inp[2, -9:-6], target[2, -27:-24])

        # print(inp[0], inp[1], inp[2])
        # print(target[0], target[1], target[2])

        return inp, target, mask, fsw, target_diff
    


class PretrainDataset(Dataset):
    def __init__(self, files, sequence_length, estimate_pose=True):
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.estimate_pose = estimate_pose
    
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])
        

        # keys: input, target, target_env, actions, ll_actions, fsw, done 
        inp = torch.tensor(data['input'])[:, :39]
        inp[:, 27:39] *= 0.08
        target = torch.tensor(data['input'])[:, :39]
        target[:, 27:39] *= 0.08
        # print(inp.shape)
        # pose = torch.tensor(data['target'])[:, :3] * torch.tensor([0.25, 1, 1/3.14])
        # randomly induce a few repeated poses to make the model learn to ignore the pose
        done_idx = data['done'].nonzero()[0][-1]

        
        # select 15 percent random indices to mask
        indices = torch.randint(0, done_idx, (int(0.15 * done_idx),))
        # augment more here incase low performance

        mask = torch.tensor(data['done'])
        mask[:] = False
        mask[indices] = True
        inp[indices, :] = 0.
        # mask
        mask, fsw = mask.unsqueeze(-1), torch.tensor(data['fsw'])
        return inp, target, mask, done_idx

# if __name__ == "__main__":
#     import pickle
#     from torch.utils.data import DataLoader
#     data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_aug27/2_obs/all_files/all_files_2.pkl'
#     with open(data_folder, 'rb') as f:
#             balanced_data = pickle.load(f)
#     files = balanced_data[:5]
#     sequence_length = 1500
#     ds = PretrainDataset(files, sequence_length)
#     train_dl = DataLoader(ds, batch_size=1, shuffle=True)

#     inp, target, mask, done_idx = next(iter(train_dl))


class PoseSceneEstimatorDataset(Dataset):
    def __init__(self, files, sequence_length, estimate_pose=True):
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.estimate_pose = estimate_pose
    
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])
        

        # keys: input, target, target_env, actions, ll_actions, fsw, done 
        inp = torch.tensor(data['input'])
        inp[:, 27:39] *= 0.08
        # print(inp.shape)
        pose = torch.tensor(data['target'])[:, :5] * torch.tensor([1., 1., 1., 0.4, 0.4])

        # print(pose[:2])
        # enable code below for noisy pose 
        # randomly induce a few repeated poses to make the model learn to ignore the pose
        done_idx = data['done'].nonzero()[0][-1]

        # repeat_count = int(np.random.uniform(0.05, 0.15) * done_idx)
        
        # if np.random.uniform() < 0.75:
        #     if np.random.uniform() < 0.75:
        #         random_indices = torch.randint(2, done_idx, (repeat_count,))
        #         pose[random_indices, :] = pose[random_indices-2, :]
        #         pose[random_indices-1, :] = pose[random_indices-2, :]
        #     else:
        #         random_indices = torch.randint(3, done_idx, (repeat_count,))
        #         pose[random_indices, :] = pose[random_indices-3, :]
        #         pose[random_indices-1, :] = pose[random_indices-3, :]
        #         pose[random_indices-2, :] = pose[random_indices-3, :]


        target = data['target_env']

        # new_target = np.zeros((target.shape[0], 24))
        # new_target[:, :4] = target[:, :4]
        # new_target[:, 4] = np.sin(target[:, 4])
        # new_target[:, 5] = np.cos(target[:, 4])
        # new_target[:, 6:8] = target[:, 5:7]
        
        # new_target[:, 8:12] = target[:, 7:11]
        # new_target[:, 12] = np.sin(target[:, 11])
        # new_target[:, 13] = np.cos(target[:, 11])
        # new_target[:, 14:16] = target[:, 12:14]

        # new_target[:, 16:20] = target[:, 14:18]
        # new_target[:, 20] = np.sin(target[:, 18])
        # new_target[:, 21] = np.cos(target[:, 18])
        # new_target[:, 22:24] = target[:, 19:21]

        # target = torch.tensor(new_target).to(torch.float32)
        # target *= torch.tensor([1, 1, 0.25, 1, 1., 1., 1., 1/1.7] * 3)

        # confidence_target = np.concatenate([np.zeros((target.shape[0], 1)), target], axis=1)

        confidence_target = np.zeros((target.shape[0], 24))
        # confidence_target = np.concatenate([np.zeros((target.shape[0], 1)), target], axis=1)

        confidence_target[:, 1:8] = target[:, :7]
        confidence_target[:, 9:16] = target[:, 7:14]
        confidence_target[:, 17:24] = target[:, 14:21]
        
        if len(target[:, 0].nonzero()[-1]) > 0:
            contact_idx = target[:, 0].nonzero()[-1][0]
            confidence_target[contact_idx:done_idx, 0] = 1.
        
        if len(target[:, 7].nonzero()[-1]) > 0:
            contact_idx = target[:, 7].nonzero()[-1][0]
            confidence_target[contact_idx:done_idx, 8] = 1.

        if len(target[:, 14].nonzero()[-1]) > 0:
            contact_idx = target[:, 14].nonzero()[-1][0]
            confidence_target[contact_idx:done_idx, 16] = 1.

        target = torch.tensor(confidence_target).to(torch.float32)
        target *= torch.tensor([1, 1, 0.25, 1, 1., 1., 1, 1/1.7] * 3)

        mask, fsw = torch.tensor(data['done']).unsqueeze(-1), torch.tensor(data['fsw'])
        # print(torch.max(pose[:, :5], dim=0))

        return torch.cat([inp, pose[:, :3]], dim=-1), target, mask, fsw, pose
    

class VeloityEstimatorDataset(Dataset):
    def __init__(self, files, sequence_length, estimate_pose=True):
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.estimate_pose = estimate_pose
    
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])

        # keys: input, target, target_env, actions, ll_actions, fsw, done 
        inp = torch.tensor(data['input'])
        inp[:, 27:39] *= 0.08
        pose = torch.tensor(data['target'])[:, :5] * torch.tensor([1., 1., 1., 0.4, 0.4])

        done_idx = data['done'].nonzero()[0][-1]
        target = pose[:, -2:].clone()
        mask, fsw = torch.tensor(data['done']).unsqueeze(-1), torch.tensor(data['fsw'])

        return torch.cat([inp], dim=-1), target, mask, fsw, pose