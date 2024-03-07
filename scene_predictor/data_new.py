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

        self.files = files
        self.sequence_length = int(sequence_length)
        self.estimate_pose = estimate_pose
    
    def __len__(self):
        return int((len(self.files)))
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        # data keys: input, target, target_env, actions, ll_actions, fsw, done 

        # input data
        inp = torch.tensor(data['input']) # project_gravity 3, joint_pos 12, joint_vel 12, torques 12

        # split inp
        projected_gravity = inp[:, :3]
        joint_pos = inp[:, 3:15]
        joint_vel = inp[:, 15:27]
        torques = inp[:, 27:39] * 0.08 # scale torques
        pose = torch.tensor(data['target'])[:, :3]
        velocity = torch.tensor(data['target'])[:, 3:5]
        inputs = {
            'projected_gravity': projected_gravity,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'torques': torques, # scale torques,
            'pose': pose,
            'velocity': velocity
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
        fsw_k = 0 
        for obs_idx in range(self.obstacles):
            true_k = (9 * obs_idx) # skip the last 2 elements (data contains: mass and friction that we do not consider in the model)
            if 'confidence' in self.output_dict:
                contact_points = target[:, true_k + 0].nonzero()
                if len(contact_points) > 0:
                    contact_idx = contact_points[-1][0]
                    # print(obs_idx, contact_idx, done_idx)
                    final_target[contact_idx:, k] = 1.
                    final_fsw[contact_idx:, k] = 1.
                k += self.output_dict['confidence']
                
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
     

        # teacher forcing
        shifted_final_target = torch.zeros((self.sequence_length, int(self.obstacles * self.output_size)))
        shifted_final_target[1:, :] = final_target.clone()[:-1, :]
        teacher_force = False
        if teacher_force:
            inputs = torch.cat([inputs, shifted_final_target], dim=-1)

        mask = torch.tensor(data['done'])[:].unsqueeze(-1)

        # print(fsw.shape)
        return inputs, final_target, mask, final_fsw, pose, target

class RealTransformerDataset(Dataset):
    def __init__(self, cfg, files, sequence_length):
        self.all_files = files
        self.sequence_length = sequence_length
        self.output_size = sum(cfg.outputs.values())
        self.obstacles = cfg.obstacles
        self.input_dict = cfg.inputs
        self.output_dict = cfg.outputs

    def __len__(self):
        return int((len(self.all_files)))

    def __getitem__(self, idx):
        data = np.load(self.all_files[idx], allow_pickle=True)

        print(list(data.keys()))
        done_idx = data['done'].nonzero()[0][-1]
        start = 0
        end = done_idx
        inp = torch.from_numpy(data['input'][start:end])
        torques_est = torch.from_numpy(data['torques_estimated'][start:end])
        inp[:, 27:39] = torques_est

        # projected_gravity = inp[:, :3] * 0.1
        joint_pos = inp[:, 3:15] * 1
        joint_vel = inp[:, 15:27] * 0.05
        torques = inp[:, 27:39] * 0.08
        pose = (torch.from_numpy(data['target'][start:end]) * torch.tensor([0.25, 1, 1/3.14]))
        # velocity = torch.from_numpy(data['body_vel'][start:end])

        inputs = {
            # 'projected_gravity': projected_gravity,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'torques': torques,
            'pose': pose,
            # 'velocity': velocity
        }

        inputs = [inputs[k] for k in self.input_dict.keys()]
        inputs = torch.cat(inputs, dim=-1)
        
        inputs = torch.cat([inputs, torch.zeros(1500 - (end-start), inputs.shape[1])], dim=0) # padding
    
        m_obstacle_data = data['obstacles'][0]
        m_obstacle_data_copy = m_obstacle_data.clone()
        # m_obstacle_data_3 = m_obstacle_data_copy[:, 3]
        # m_obstacle_data_4 = m_obstacle_data_copy[:, 4]

        m_obstacle_data[:, 3] = m_obstacle_data_copy[:, 4]
        m_obstacle_data[:, 4] = m_obstacle_data_copy[:, 3]

        im_obstacle_data = data['obstacles'][2]
        im_obstacle_data_copy = im_obstacle_data.clone()
        # im_obstacle_data_3 = im_obstacle_data_copy[:, 3]
        # im_obstacle_data_4 = im_obstacle_data_copy[:, 4]
        im_obstacle_data[:, 3] = im_obstacle_data_copy[:, 4]
        im_obstacle_data[:, 4] = im_obstacle_data_copy[:, 3]
        
        
        
        target = torch.from_numpy(np.concatenate([m_obstacle_data, im_obstacle_data], axis=1))

        mask = torch.zeros((1500, 1), dtype=torch.bool)
        mask[:end, :] = True
        
        return inputs, target, mask, pose
    
    def calculate_box_corners(x, y, theta, w, h):
        """
        Calculate the bottom-left and top-right corners of a 2D box given its center,
        width, height, and rotation angle.
        
        Parameters:
        - x, y: Coordinates of the box's center.
        - theta: Rotation angle of the box in radians.
        - w, h: Width and height of the box.
        
        Returns:
        - (bl_x, bl_y, tr_x, tr_y): Coordinates of the bottom-left and top-right corners.
        """
        def rotate_point(px, py, theta):
            """Rotate a point by theta around the origin."""
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            x_rotated = px * cos_theta - py * sin_theta
            y_rotated = px * sin_theta + py * cos_theta
            return x_rotated, y_rotated

        # Half dimensions
        half_w, half_h = w / 2, h / 2
        
        # Calculate offsets for corners in local box coordinates
        offsets = [(-half_w, -half_h), (half_w, half_h)]
        
        # Rotate offsets and translate to world coordinates
        corners = [rotate_point(ox, oy, theta) for ox, oy in offsets]
        corners_world = [(x + cx, y + cy) for cx, cy in corners]
        
        # Bottom-left and top-right corners
        bl_x, bl_y = corners_world[0]
        tr_x, tr_y = corners_world[1]
        
        return bl_x, bl_y, tr_x, tr_y

    

class VelocityTransformerDataset(Dataset):
    def __init__(self, cfg, files, sequence_length):
        
        self.output_size = sum(cfg.outputs.values())
        self.input_dict = cfg.inputs
        self.output_dict = cfg.outputs

        self.files = files
        self.sequence_length = int(sequence_length)
    
    def __len__(self):
        return int((len(self.files)))
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        # data keys: input, target, target_env, actions, ll_actions, fsw, done 

        # input data
        inp = torch.tensor(data['input']) # project_gravity 3, joint_pos 12, joint_vel 12, torques 12

        # split inp
        projected_gravity = inp[:, :3]
        joint_pos = inp[:, 3:15]
        joint_vel = inp[:, 15:27]
        torques = inp[:, 27:39] * 0.08 # scale torques
        pose = torch.tensor(data['target'])[:, :3]
        vel = torch.tensor(data['target'])[:, 3:5]
        
        inputs = {
            'projected_gravity': projected_gravity,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'torques': torques, # scale torques,
            'pose': pose
        }

        inputs = [inputs[k] for k in self.input_dict.keys()]
        inputs = torch.cat(inputs, dim=-1)

        done_idx = data['done'][:].nonzero()[0][-1]
        
        # target data
        target = vel
        mask = torch.tensor(data['done'])[:].unsqueeze(-1)

        return inputs, target, mask
    
if __name__ == '__main__':

    import pickle
    from runner_config import RunCfg
    import random

    cfg = RunCfg.transformer.data_params

    # data_files = [sorted(glob(f'/common/home/dm1487/Downloads/sep15/2/*.npz'))[-1]]

    with open('/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24/balanced/train_1.pkl', 'rb') as f:
        files = pickle.load(f)

    random.shuffle(files)
    dataset = TransformerDataset(cfg, files, 1500)
    for i in range(50):
        print("#")
        _ = dataset[i]
    