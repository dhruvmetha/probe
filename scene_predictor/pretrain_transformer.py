from data import PretrainDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from model import PretrainTransformer
from tqdm import tqdm
from glob import glob
from visualization import get_visualization
import torch
import pickle
import random
import numpy as np
import wandb

# wandb.init(project='pretain_transformer', entity='dm1487')

class PretrainPredictor:
    def __init__(self):
        
        self.optimizer = None
        self.scheduler = None
        self.sequence_length = 1500
        self.device = 'cuda:0'

        self.model = PretrainTransformer(input_size=39, output_size=39, embed_size=512, hidden_size=2048, num_heads=2, max_sequence_length=self.sequence_length, num_layers=2)
        self.src_mask = torch.ones(self.sequence_length, self.sequence_length).bool().to(self.device)

    def loss_fn(self, out, targ):
        k = 0 + 0
        loss = F.mse_loss(out[:, :, :], targ[:, :, :], reduction='none')
        loss = torch.sum(loss, dim=-1).unsqueeze(-1)
        return loss

    def train(self, dl, val_dl, save_folder, print_every=50, eval_every=250):
        self.model.train()
        train_loss, val_loss, val_ctr = 0, 0, 0
        for i, (inp, target, mask, done_idx) in tqdm(enumerate(dl)):
            inp = inp.to(self.device)
            targ = target.to(self.device)
            mask = mask.to(self.device)

            out = self.model(inp)
            loss = self.loss_fn(out, targ)
            loss = torch.sum(loss*mask)/torch.sum(mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
          
            if (i+1) % print_every == 0:
                print(f'step {i+1}:', train_loss/print_every)
                # wandb.log({
                #     'train_loss': train_loss/print_every,
                # })
                train_loss = 0

            if (i+1) % eval_every == 0:
                new_val_loss = self.validate(val_dl, save_folder)
                val_loss += new_val_loss
                val_ctr += 1
                # wandb.log({
                #     'val loss': new_val_loss,
                # })
                self.model.train()
        self.scheduler.step(val_loss/val_ctr)
        return val_loss/val_ctr

    def validate(self, dl, save_folder):
        vis_path = f'{save_folder}/viz'
        self.model.eval()
        with torch.inference_mode():
            val_loss = 0
            for i, (inp, target, mask, done_idx) in tqdm(enumerate(dl)):
                inp = inp.to(self.device)
                targ = target.to(self.device)
                mask = mask.to(self.device)
                out = self.model(inp)
                loss = self.loss_fn(out, targ)
                loss = torch.sum(loss*mask)/torch.sum(mask)
                val_loss += loss.item()
            print(f'validation loss: {val_loss/(i+1)}')
        
        return val_loss/(i+1)

    def load_model(self, model_path, device='cuda:0'):
        self.model = torch.load(model_path)
        self.device = device
        print('model loaded')

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
        
        save_path = f'{save_folder}/checkpoints'

        with open(data_folder, 'rb') as f:
            balanced_data = pickle.load(f)
        # balanced_data = glob(data_folder)
        random.shuffle(balanced_data)
        print('# trajectories:', len(balanced_data))
        # return
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
            if len(training_files) > 64000:
                sub_training_files = random.sample(training_files, 64000)
            else:
                sub_training_files = training_files
            
            if len(val_files) > 4000:
                sub_val_files = random.sample(val_files, 4000)
            else:
                sub_val_files = val_files

            train_ds = PretrainDataset(files=sub_training_files, sequence_length=self.sequence_length, estimate_pose=True)
            train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
            
            val_ds = PretrainDataset(files=sub_val_files, sequence_length=self.sequence_length, estimate_pose=True)
            val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=True)

            val_loss = self.train(train_dl, val_dl, save_folder, print_every=print_every, eval_every=eval_every)

            PATH = f'{save_path}/model_state_dict{epoch}.pt'
            torch.save(self.model.state_dict(), PATH)
            

if __name__ == '__main__':
    from torch import nn

    # load_pretrained_model = True
    # device = 'cpu'
    # se = PretrainPredictor()
    # if load_pretrained_model:
    #     model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/pretrain_2_obs/2023-08-29_20-33-05/checkpoints/model_11.pt'
    #     se.load_model(model_path, device=device)

    # # for i in se.model.children():
    # #     print(i)

    # # se.model.encoder  

    # with torch.no_grad():
    #     se.model.activation = nn.Identity()
    #     se.model.out = nn.Identity()
    
    # print(se.model.out)
    # print(se.model.activation)

    # dummy = torch.randn(1, 1500, 39).to('cuda:0')
    # out = se.model(dummy)
    # print(out.shape)
    # for i in se.model.children():
    #     print(i)
    # for i in se.model.children():
    #     print(i)
    

    from datetime import datetime
    import os

    # data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/1_obs/8/*/*.npz'
    save_folder = f'/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/pretrain_2_obs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    # data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/test/0/all_files.pkl'
    data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_aug30/2_obs/all_files/all_files_1.pkl'
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(f'{save_folder}/checkpoints')
        os.makedirs(f'{save_folder}/viz')
    

    load_model = False
    device = 'cuda:0'
    pe = PretrainPredictor()
    if load_model:
        model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/pretrain_2_obs/2023-08-25_12-08-23/checkpoints/model_2.pt'
        pe.load_model(model_path, device=device)

    with open(f'{save_folder}/info.txt', 'w') as f:
        f.write('pretrain transformer, all data, 2 layers, 2 heads')
    
    pe.runner(data_folder, save_folder, epochs=50, train_test_split=0.9, train_batch_size=64, val_batch_size=64, learning_rate=1e-4, device='cuda:0', print_every=50, eval_every=500)
