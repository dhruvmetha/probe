from data import TransformerDataset, PoseEstimatorDataset, PoseDiffEstimatorDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from model import MiniTransformer
from tqdm import tqdm
from glob import glob
from visualization import get_visualization
import torch
import pickle
import random
import numpy as np

class PoseEstimator:
    def __init__(self):
        
        self.optimizer = None
        self.scheduler = None
        self.sequence_length = 1500
        self.device = 'cuda:0'

        self.model = MiniTransformer(input_size=29, output_size=5, embed_size=128, hidden_size=2048, num_heads=2, max_sequence_length=self.sequence_length, num_layers=6, estimate_pose=True)
        self.src_mask = torch.triu(torch.ones(self.sequence_length, self.sequence_length), diagonal=1).bool().to(self.device)

        self.saved_videos = 0

    def loss_fn(self, out, targ):

        loss_pose = F.mse_loss(out, targ, reduction='none')

        # loss_xy = F.mse_loss(out[:, :, :2], targ[:, :, :2], reduction='none')
        # loss_xy = torch.sum(loss_xy, dim=-1).unsqueeze(-1)
        
        # loss_theta = F.mse_loss(out[:, :, 2:3], targ[:, :, 2:3], reduction='none')
        # loss_theta = torch.sum(loss_theta, dim=-1).unsqueeze(-1)

        # loss_vel = F.mse_loss(out[:, :, 3:], targ[:, :, 3:], reduction='none')
        # loss_vel = torch.sum(loss_vel, dim=-1).unsqueeze(-1)
        
        # loss_pose = torch.sum(loss_pose, dim=-1).unsqueeze(-1)
    
        return loss_pose # loss_xy, loss_theta, loss_vel

    def train(self, dl, val_dl, test_dl, save_folder, print_every=50, eval_every=250):
        self.model.train()
        train_loss, val_loss = 0, 0
        for i, (inp, targ, mask, fsw, targ_diff) in tqdm(enumerate(dl)):
            inp = inp.to(self.device)
            targ = targ.to(self.device) 
            targ_diff = targ_diff.to(self.device)
            mask = mask.to(self.device)
            out = self.model(inp, src_mask=self.src_mask)

            # loss_xy, loss_theta, loss_vel = self.loss_fn(out, targ)
            # loss = torch.sum(loss_xy*mask)/torch.sum(mask) + torch.sum(loss_theta*mask)/torch.sum(mask) + torch.sum(loss_vel*mask)/torch.sum(mask)
            # out[:, :, :3] /= 10
            out = targ.detach() + out
            loss_pose = self.loss_fn(out, targ)
            loss = torch.sum(loss_pose*mask)/torch.sum(mask)
            # out *= torch.tensor([[[0.25, 1, 1/3.14, 1/0.4, 1/0.4]]], device=self.device)
            
            # loss_pose = self.loss_fn(out, targ_diff)
            # loss = torch.sum(loss_pose*mask)/torch.sum(mask)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            if (i+1) % print_every == 0:
                print(f'step {i+1}:', train_loss/print_every)
                train_loss = 0

            if (i+1) % eval_every == 0:
                self.test(test_dl, save_folder)
                val_loss += self.validate(val_dl, save_folder)
                self.model.train()
            
            
        return val_loss

    def validate(self, dl, save_folder):
        vis_path = f'{save_folder}/viz'
        self.model.eval()
        with torch.inference_mode():
            val_loss = 0
            for i, (inp, targ, mask, fsw, targ_diff) in tqdm(enumerate(dl)):
                inp = inp.to(self.device)
                # inp_pose = inp[:, :, -5:].clone().to(self.device) 
                targ = targ.to(self.device)
                targ_diff = targ_diff.to(self.device)
                mask = mask.to(self.device)
                out = self.model(inp, src_mask=self.src_mask)

                # out[:, :, :3] /= 10
                out = targ.detach() + out
                loss_pose = self.loss_fn(out, targ)
                loss = torch.sum(loss_pose*mask)/torch.sum(mask)

                # loss_xy, loss_theta, loss_vel = self.loss_fn(out, targ)
                # loss = torch.sum(loss_xy*mask)/torch.sum(mask) + torch.sum(loss_theta*mask)/torch.sum(mask) + torch.sum(loss_vel*mask)/torch.sum(mask)

                # loss_pose = self.loss_fn(out, targ_diff)
                # loss = torch.sum(loss_pose*mask)/torch.sum(mask)
                                 
                val_loss += loss.item()

                if i < 5:
                    final_out = out.detach()
                    final_out[:, :, :3] /= 100
                    final_out *= torch.tensor([[[0.25, 1, 1/3.14, 1/0.4, 1/0.4]]], device=self.device)
                    targ[:, :, :3] /= 100
                    targ *= torch.tensor([[[0.25, 1, 1/3.14, 1/0.4, 1/0.4]]], device=self.device)

                    self.save_visualization(inp, targ, final_out, fsw, mask, vis_path)
            print(f'validation loss: {val_loss/(i+1)}')
        
        return val_loss
    
    def test(self, dl, save_folder):
        vis_path = f'{save_folder}/viz_test'
        self.model.eval()
        with torch.inference_mode():
            val_loss = 0
            for i, (inp, targ, mask, fsw, targ_diff) in tqdm(enumerate(dl)):
                if i < 2:
                    inp = inp.to(self.device)
                    targ = targ.to(self.device)
                    targ_diff = targ_diff.to(self.device)
                    mask = mask.to(self.device)

                    inp_fb = inp.clone()
                    inp_fb[:, 1:, 24:] = 0.
                    for j in tqdm(range(inp.shape[1] - 1)):
                        out = self.model(inp_fb, src_mask=self.src_mask)
                        out[:, :, :3] /= 100
                        out *= torch.tensor([[[0.25, 1, 1/3.14, 1/0.4, 1/0.4]]], device=self.device)
                        inp_fb[:, j+1:j+2, 24:] = inp_fb[:, j:j+1, 24:] + ((out.detach()[:, j:j+1, :]))

                    new_inp_fb = inp_fb.clone()
                    new_inp_fb[:, :, 24:] *= torch.tensor([[[1/0.25, 1, 3.14, 0.4, 0.4]]],  device=self.device)
                    new_inp_fb[:, :, 24:27] *= 100
                    loss_pose = self.loss_fn(new_inp_fb[:, :, 24:], targ[:, :, :])
                    loss = torch.sum(loss_pose*mask)/torch.sum(mask)
                                    
                    val_loss += loss.item()
                    
                    final_out = inp_fb[:, :, 24:]
                    targ[:, :, :3] /= 100
                    targ *= torch.tensor([[[0.25, 1, 1/3.14, 1/0.4, 1/0.4]]], device=self.device)
                    self.save_visualization(inp, targ, final_out, fsw, mask, vis_path)

                else:
                    print(f'test loss: {val_loss/(i)}')
                    break
        
        return val_loss

    def save_visualization(self, inp, targ, out, fsw, mask, save_path):
        k = 0
        patches = []
        for step in range(inp.shape[1]):
            if mask[0, step, 0]:
                patch_set = get_visualization(0, targ[:, step, :]*torch.tensor([1/0.25, 1, 3.14, 0.4, 0.4], device=self.device), fsw[:, step, :], out[:, step, :]* torch.tensor([1/0.25, 1, 3.14, 0.4, 0.4], device=self.device), fsw[:, step, :], fsw[:, step, :].squeeze(1), estimate_pose=True)
                patches.append(patch_set)
        with open(f'{save_path}/plot_{self.saved_videos}.pkl', 'wb') as f:
                pickle.dump(patches, f)
        self.saved_videos += 1
    
    def load_model(self, model_path, device='cuda:0'):
        self.model = torch.jit.load(model_path)
        self.device = device

    def predict(self, data):
        self.model.eval()
        with torch.inference_mode():
            inp = torch.tensor(data['input']).to(self.device)
            targ = torch.tensor(data['target']).to(self.device)
            mask = torch.tensor(data['done']).unsqueeze(-1).to(self.device)
            out = self.model(inp, src_mask=self.src_mask)
            loss_pose = self.loss_fn(out, targ)
            loss = torch.sum(loss_pose*mask)/torch.sum(mask)
            print(f'validation loss: {loss}')

        return out.detach().cpu().numpy()
        

    def runner(self, data_folder, save_folder, epochs=100, train_test_split=0.9, train_batch_size=32, val_batch_size=32, learning_rate=1e-4, device='cuda:0', print_every=50, eval_every=250):
        

        with open(data_folder, 'rb') as f:
            balanced_data = pickle.load(f)
        # balanced_data = glob(data_folder)
        # balanced_data
        random.shuffle(balanced_data)
        print('# trajectories:', len(balanced_data))
        # return

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            os.makedirs(f'{save_folder}/checkpoints')
            os.makedirs(f'{save_folder}/viz')
            os.makedirs(f'{save_folder}/viz_test')
        save_path = f'{save_folder}/checkpoints'

        self.device = device
        self.model = self.model.to(self.device)
        self.src_mask = self.src_mask.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        num_train_envs = int(len(balanced_data) * train_test_split)
        train_idxs = np.arange(0, num_train_envs).astype(int).tolist()
        val_idxs = np.arange(num_train_envs, len(balanced_data)).astype(int).tolist()
        training_files = [balanced_data[i] for i in train_idxs]
        val_files = [balanced_data[i] for i in val_idxs]

        for epoch in range(epochs):
            if len(training_files) > 96000:
                sub_training_files = random.sample(training_files, 96000)
            else:
                sub_training_files = training_files
            
            if len(val_files) > 5000:
                sub_val_files = random.sample(val_files, 5000)
            else:
                sub_val_files = val_files

            train_ds = PoseDiffEstimatorDataset(files=sub_training_files, sequence_length=self.sequence_length, estimate_pose=True)
            train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
            
            val_ds = PoseDiffEstimatorDataset(files=sub_val_files, sequence_length=self.sequence_length, estimate_pose=True)
            val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=True)

            # val_ds = PoseDiffEstimatorDataset(files=sub_val_files, sequence_length=self.sequence_length, estimate_pose=True)
            test_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

            val_loss = self.train(train_dl, val_dl, test_dl, save_folder, print_every=print_every, eval_every=eval_every)
            self.scheduler.step(val_loss)

            torch.jit.save(torch.jit.script(self.model), f'{save_path}/model_{epoch}.pt')
                

if __name__ == '__main__':
    from datetime import datetime
    import os

    # data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/1_obs/8/*/*.npz'
    save_folder = f'/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_1_obs_pe/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    
    # data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data/1_obs/0/all_files.pkl'
    data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data/2_obs/11/all_files.pkl'
    load_model = False
    device = 'cuda:0'
    pe = PoseEstimator()
    if load_model:
        # model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_1_obs_pe/2023-08-24_15-55-05/checkpoints/model_1.pt'
        model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_1_obs_pe/2023-08-25_12-41-21/checkpoints/model_0.pt'
        pe.load_model(model_path, device=device)

    pe.runner(data_folder, save_folder, epochs=40, train_test_split=0.9, train_batch_size=16, val_batch_size=16, learning_rate=1e-4, device=device, print_every=50, eval_every=250)
