import torch
from scene_predictor.model import MiniTransformer
from scene_predictor.visualization import get_visualization 
import pickle


class PoseInference:
    def __init__(self, model_path='', sequence_length=750, device='cuda:0'):
        self.device = device
        output_size = 24 + 6 + 21 # 27
        self.sequence_length = sequence_length
        try:
            self.model = torch.jit.load(model_path)
        except:
            self.model = MiniTransformer(input_size=27, output_size=output_size, embed_size=512, hidden_size=2048, num_heads=4, max_sequence_length=750, num_layers=4, estimate_pose=True)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, x):
        with torch.inference_mode():
            x = x.to(self.device)
            # src_mask = torch.triu(torch.ones(self.sequence_length-1, self.sequence_length-1) * float('-inf'), diagonal=1).to(self.device)
            src_mask = torch.triu(torch.ones(self.sequence_length-1, self.sequence_length-1), diagonal=1).bool().to(device)
            output = self.model(x, src_mask)
            return output.detach()
        
    def infer_and_plot(self, data, save_path):
        patches = []
        inp_obs = torch.tensor(data['input'][1:sequence_length, :])
        pose = torch.tensor(data['target'][1:sequence_length, :6]).to(device).unsqueeze(0)
        mask, fsw = torch.tensor(data['done'][1:sequence_length]).unsqueeze(-1), torch.tensor(data['fsw'][1:sequence_length, :]).to(device).unsqueeze(0)
        projected_gravity = inp_obs[:, :3]
        joint_pos = inp_obs[:, 18:30]
        joint_vel = inp_obs[:, 30:42]
        final_inp_pose = torch.cat([projected_gravity, joint_pos, joint_vel], dim=-1).unsqueeze(0)
        output = self.predict(final_inp_pose)

        for step in range(pose.shape[1]):
            if mask[step, 0]:
                ## only pose
                patch_set = get_visualization(0, pose[:, step, :6]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), pose[:, step, 6:], output[:, step, :6]* torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), output[:, step, 6:], fsw[:, step, :].squeeze(1), estimate_pose=True)
                patches.append(patch_set)
        # save the patches using pickle
        with open(save_path, 'wb') as f:
            pickle.dump(patches, f)

class ObstacleInference:
    def __init__(self, model_path='', sequence_length=750, device='cuda:0'):
        self.device = device
        output_size = 24 + 6 + 21 # 27
        self.sequence_length = sequence_length
        # self.model = MiniTransformer(input_size=12+27+6, output_size=output_size, embed_size=512, hidden_size=2048, num_heads=4, max_sequence_length=750, num_layers=6, estimate_pose=False)
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = torch.jit.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, x):
        with torch.inference_mode():
            x = x.to(self.device)
            # src_mask = torch.triu(torch.ones(self.sequence_length-1, self.sequence_length-1) * float('-inf'), diagonal=1).to(self.device)
            src_mask = torch.triu(torch.ones(self.sequence_length-1, self.sequence_length-1), diagonal=1).bool().to(device)
            output = self.model(x, src_mask)
            return output.detach()
        
    
    # def infer_and_plot(self, data, save_path):
    #     patches = []
    #     torques = torch.tensor(data['torques'][1:self.sequence_length, :])
    #     inp_obs = torch.tensor(data['input'][1:sequence_length, :])
    #     pose = torch.tensor(data['target'][1:sequence_length, :6]).to(device).unsqueeze(0)
    #     priv_info = torch.tensor(data['target'][1:sequence_length, 6:]).to(device).unsqueeze(0)
    #     mask, fsw = torch.tensor(data['done'][1:sequence_length]).unsqueeze(-1), torch.tensor(data['fsw'][1:sequence_length, :]).to(device).unsqueeze(0)
    #     projected_gravity = inp_obs[:, :3]
    #     joint_pos = inp_obs[:, 18:30]
    #     joint_vel = inp_obs[:, 30:42]
    #     final_inp_pose = torch.cat([torques, projected_gravity, joint_pos, joint_vel, pose], dim=-1).unsqueeze(0)
    #     output = pose_inference.predict(final_inp_pose)

    #     for step in range(pose.shape[1]):
    #         if mask[step, 0]:
    #             ## only pose
    #             patch_set = get_visualization(0, pose[:, step, :6]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), priv_info[:, step, :]*torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device), pose[:, step, :6]* torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), output[:, step, :]*torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device), fsw[:, step, :].squeeze(1), estimate_pose=True)
    #             patches.append(patch_set)
    #     # save the patches using pickle
    #     with open(save_path, 'wb') as f:
    #         pickle.dump(patches, f)

# this class does inference sequentially first PoseInference and then ObstacleInference
class FullInference:
    def __init__(self, pose_model_path='', obstacle_model_path='', sequence_length=750, device='cuda:0'):
        self.device = device
        self.sequence_length = sequence_length
        self.pose_inference = PoseInference(model_path=pose_model_path, sequence_length=sequence_length, device=device)
        self.obstacle_inference = ObstacleInference(model_path=obstacle_model_path, sequence_length=sequence_length, device=device)
    
    def infer_and_plot(self, data, save_path):
        patches = []
        inp_obs = torch.tensor(data['input'][1:sequence_length, :])
        projected_gravity = inp_obs[:, :3]
        joint_pos = inp_obs[:, 18:30]
        joint_vel = inp_obs[:, 30:42]
        final_inp_pose = torch.cat([projected_gravity, joint_pos, joint_vel], dim=-1).unsqueeze(0)
        pred_pose = self.pose_inference.predict(final_inp_pose)

        torques = torch.tensor(data['torques'][1:self.sequence_length, :])
        touch_inp_pose = torch.cat([torques, projected_gravity, joint_pos, joint_vel, pred_pose.squeeze(0).cpu()], dim=-1).unsqueeze(0)
        pred_priv_info = self.obstacle_inference.predict(touch_inp_pose)

        priv_info = torch.tensor(data['target'][1:sequence_length, 6:]).to(device).unsqueeze(0)
        pose = torch.tensor(data['target'][1:sequence_length, :6]).to(device).unsqueeze(0)
        mask, fsw = torch.tensor(data['done'][1:sequence_length]).unsqueeze(-1), torch.tensor(data['fsw'][1:sequence_length, :]).to(device).unsqueeze(0)

        for step in range(pose.shape[1]):
            if mask[step, 0]:
                ## only pose
                patch_set = get_visualization(0, pose[:, step, :6]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), priv_info[:, step, :], pred_pose[:, step, :6]* torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), pred_priv_info[:, step, :]*torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device), fsw[:, step, :].squeeze(1), estimate_pose=False)
                # print(pose[:, step, :2], pred_pose[:, step, :2])
                patches.append(patch_set)
        # save the patches using pickle
        with open(save_path, 'wb') as f:
            pickle.dump(patches, f)

if __name__ == '__main__':
    # pass
    from glob import glob
    import numpy as np
    from pathlib import Path
    import random
    from tqdm import tqdm
    ###### works #######
    POSE_FOLDER = Path(f'./scene_predictor/results_pose/transformer_750_2048/2023-06-03_22-11-43')
    PRIV_INFO_FOLDER = Path(f'./scene_predictor/results_priv_info/transformer_750_2048/2023-06-03_22-26-56')
    # PRIV_INFO_FOLDER = Path(f'./scene_predictor/results/transformer_750_2048/2023-05-20_21-12-37')
    device = 'cuda:0'
    sequence_length = 750
    full_inference = FullInference(pose_model_path=str(POSE_FOLDER/'checkpoints/model_29.pt'), obstacle_model_path=str(PRIV_INFO_FOLDER/'checkpoints/model_23.pt'), sequence_length=sequence_length, device=device)

    #### works ########


    # ###### works #######
    # POSE_FOLDER = Path(f'./scene_predictor/results/transformer_750_2048/2023-05-20_00-36-45')
    # PRIV_INFO_FOLDER = Path(f'./scene_predictor/results_priv_info/transformer_750_2048/2023-05-23_02-09-44')
    # # PRIV_INFO_FOLDER = Path(f'./scene_predictor/results/transformer_750_2048/2023-05-20_21-12-37')
    # device = 'cuda:0'
    # sequence_length = 750
    # full_inference = FullInference(pose_model_path=str(POSE_FOLDER/'checkpoints/model_96.pt'), obstacle_model_path=str(PRIV_INFO_FOLDER/'checkpoints/model_91.pt'), sequence_length=sequence_length, device=device)
    # #### works ########

    # POSE_FOLDER = Path(f'./scene_predictor/results_pose/transformer_750_2048/2023-05-23_16-59-30')
    # PRIV_INFO_FOLDER = Path(f'./scene_predictor/results_priv_info/transformer_750_2048/2023-05-23_19-46-22')
    
    SAVE_FOLDER = Path(f'./scene_predictor/decoupled_inference/5')
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
    PLOT_FOLDER = SAVE_FOLDER/'plots_inference'
    PLOT_FOLDER.mkdir(exist_ok=True)
    ctr = len(glob(str(PLOT_FOLDER/f'plots_*.pkl')))

    # files = sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data_1/only_random_seed_test_6_single_trajectories/*/*.npz'))
    files = glob('/common/users/dm1487/legged_manipulation_data_store/temp/play_obs_data_2_single_trajectories/*/*.npz')
    random.shuffle(files)
    for idx, file in tqdm(enumerate(files[:5])):
        data = np.load(file)
        full_inference.infer_and_plot(data, PLOT_FOLDER/f'plots_{idx+ctr}.pkl')
    