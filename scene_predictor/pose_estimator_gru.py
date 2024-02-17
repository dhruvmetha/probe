from data import TransformerDataset, PoseEstimatorDataset, PoseEstimatorGRUDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from model import GRUPose
from tqdm import tqdm
from glob import glob
from visualization import get_visualization
import torch
import pickle
import random
import numpy as np

class GRUPoseEstimator:
    def __init__(self):
        
        self.optimizer = None
        self.scheduler = None
        self.sequence_length = 1500
        self.hidden_size = 512
        self.short_seq = 100
        self.num_layers = 2
        self.device = 'cuda:0'

        self.model = GRUPose(input_size=24, output_size=5, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.2)
        
        self.saved_videos = 0

    def loss_fn(self, out, targ):
        loss_pose = F.mse_loss(out[:, :, :5], targ[:, :, :5], reduction='none')
        loss_pose = torch.sum(loss_pose, dim=-1).unsqueeze(-1)
        return loss_pose

    def train(self, dl, val_files, save_folder, val_batch_size=32, print_every=50, eval_every=250):
        self.model.train()
        train_loss, val_loss = 0, 0
        for i, (inp, targ, mask, fsw, targ_diff) in tqdm(enumerate(dl)):
            inp = inp.to(self.device)
            targ = targ.to(self.device)
            targ_diff = targ_diff.to(self.device)
            mask = mask.to(self.device)
            # out = torch.zeros_like(targ).to(self.device)
            hidden = torch.zeros(self.num_layers, inp.shape[0],self.hidden_size).to(self.device)
            start = 0
            end = 1500
            seq_len = self.short_seq
            seq_loss = 0
            while start < end:
                gru_inp = inp[:, start:start+seq_len, :]
                gru_targ = targ[:, start:start+seq_len, :]
                gru_targ_diff = targ_diff[:, start:start+seq_len, :]
                gru_mask = mask[:, start:start+seq_len, :]
                gru_out, fh, _ = self.model(gru_inp, hidden)

                loss_pose = self.loss_fn(gru_out, gru_targ_diff)
                
                if torch.sum(gru_mask) == 0:
                    loss = torch.sum(loss_pose*gru_mask)
                else:
                    loss = torch.sum(loss_pose*gru_mask)/torch.sum(gru_mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                hidden[:, :, :] = fh.detach()
                
                start += seq_len

            if (i+1) % print_every == 0:
                print(f'step {i+1}:', train_loss/print_every)
                train_loss = 0

            if (i+1) % eval_every == 0:
                val_loss += self.validate(val_files, val_batch_size, save_folder)
                self.model.train()
            
        return val_loss

    def validate(self, val_files, val_batch_size, save_folder):
        vis_path = f'{save_folder}/viz'
        self.model.eval()

        if len(val_files) > 6000:
            sub_val_files = random.sample(val_files, 6000)
        else:
            sub_val_files = val_files
            
        ds = PoseEstimatorGRUDataset(files=sub_val_files, sequence_length=self.sequence_length)
        dl = DataLoader(ds, batch_size=val_batch_size, shuffle=True)
        
        with torch.inference_mode():
            val_loss = 0
            for i, (inp, targ, mask, fsw, targ_diff) in tqdm(enumerate(dl)):
                inp = inp.to(self.device)
                targ = targ.to(self.device)
                targ_diff = targ_diff.to(self.device)
                mask = mask.to(self.device)
                out = torch.zeros_like(targ).to(self.device)
                hidden = torch.zeros(self.num_layers, inp.shape[0], self.hidden_size).to(self.device)
                start = 0
                end = 1500
                seq_len = self.short_seq
                while start < end:
                    gru_inp = inp[:, start:start+seq_len, :]
                    gru_targ = targ[:, start:start+seq_len, :]
                    gru_targ_diff = targ_diff[:, start:start+seq_len, :]
                    gru_mask = mask[:, start:start+seq_len, :]
                    gru_out, fh, _ = self.model(gru_inp, hidden)

                    loss_pose = self.loss_fn(gru_out, gru_targ_diff)

                    if torch.sum(gru_mask) == 0:
                        loss = torch.sum(loss_pose*gru_mask)
                    else:
                        loss = torch.sum(loss_pose*gru_mask)/torch.sum(gru_mask)

                    val_loss += loss.item()
                    hidden[:, :, :] = fh.detach()
                    # out[:, start:start+seq_len, :] = gru_out.detach()
                    # print(out)
                    # print(torch.cumsum(gru_out.detach(), dim=1))
                    # print(torch.cumsum(gru_out.detach(), dim=1).shape)
                    out[:, start:start+seq_len, :] = torch.cumsum(gru_out.detach(), dim=1)
                    if start > 0:
                        out[:, start:start+seq_len, :] += out[:, start-1:start, :]
                    start += seq_len

                if i < 5:
                    out[:, :, :3] /= 10
                    out[:, :, 3:] /= 1
                    self.save_visualization(inp, targ, out, fsw, mask, vis_path)
            print(f'validation loss: {val_loss/(i+1)}')
        
        return val_loss
    
    def save_visualization(self, inp, targ, out, fsw, mask, save_path):
        k = 0
        patches = []
        for step in range(inp.shape[1]):
            if mask[0, step, 0]:
                # patch_set = get_visualization(0, targ[:, step, :6]*torch.tensor([1/0.25, 1, 3.14, 0.4, 0.4], device=self.device), fsw[:, step, :], out[:, step, :]* torch.tensor([1/0.25, 1, 3.14, 0.4, 0.4], device=self.device), fsw[:, step, :], fsw[:, step, :].squeeze(1), estimate_pose=False)
                patch_set = get_visualization(0, targ[:, step, :6]*torch.tensor([1/0.25, 1, 3.14, 0.4, 0.4], device=self.device), fsw[:, step, :], out[:, step, :], fsw[:, step, :], fsw[:, step, :].squeeze(1), estimate_pose=False)

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
        

        # balanced_data = glob(data_folder)
        with open(data_folder, 'rb') as f:
            balanced_data = pickle.load(f)
        random.shuffle(balanced_data)
        print('# trajectories:', len(balanced_data))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            os.makedirs(f'{save_folder}/checkpoints')
            os.makedirs(f'{save_folder}/viz')
        save_path = f'{save_folder}/checkpoints'

        self.device = device
        self.model = self.model.to(self.device)
        # self.src_mask = self.src_mask.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        num_train_envs = int(len(balanced_data) * train_test_split)
        train_idxs = np.arange(0, num_train_envs).astype(int).tolist()
        val_idxs = np.arange(num_train_envs, len(balanced_data)).astype(int).tolist()
        training_files = [balanced_data[i] for i in train_idxs]
        val_files = [balanced_data[i] for i in val_idxs]

        for epoch in range(epochs):
            if len(training_files) > 200000:
                sub_training_files = random.sample(training_files, 200000)
            else:
                sub_training_files = training_files
            

            train_ds = PoseEstimatorGRUDataset(files=sub_training_files, sequence_length=self.sequence_length)
            train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
            
            

            val_loss = self.train(train_dl, val_files, save_folder, val_batch_size=val_batch_size, print_every=print_every, eval_every=eval_every)
            self.scheduler.step(val_loss)

            torch.jit.save(torch.jit.script(self.model), f'{save_path}/model_{epoch}.pt')
                

if __name__ == '__main__':
    from datetime import datetime
    import os

    # data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/1_obs/8/*/*.npz'
    save_folder = f'/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_1_obs_pe_gru/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/test/0/all_files.pkl'
    

    pe = GRUPoseEstimator()
    pe.runner(data_folder, save_folder, epochs=100, train_test_split=0.9, train_batch_size=250, val_batch_size=1024, learning_rate=1e-4, device='cuda:0', print_every=50, eval_every=250)


