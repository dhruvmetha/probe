from data_new import VelocityTransformerDataset
from model import MiniTransformer
from runner_config import RunCfg
from torch.utils.data import DataLoader
from params_proto import PrefixProto
from visualization import get_visualization, get_real_visualization
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import glob
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import sys

FFwriter = animation.FFMpegWriter

class VelocityTransformerModel:
    def __init__(self, cfg: RunCfg.velocity_model, data_source, save_folder, directory, device='cuda:0'):

        self.cfg = cfg

        self.model_params = cfg.model_params
        self.train_params = cfg.train_params
        self.data_params = cfg.data_params
        self.loss_scales = cfg.loss_scales
        self.logging = cfg.logging

        self.input_args = cfg.data_params.inputs.keys()
        self.input_size = sum(cfg.data_params.inputs.values())
        self.output_args = cfg.data_params.outputs.keys()
        self.output_size = sum(cfg.data_params.outputs.values())

        # set the device
        self.device = device

        # initialize the model
        self.model = MiniTransformer(input_size=self.input_size, 
                                     output_size=self.output_size,
                                     embed_size=self.model_params.embed_size, 
                                     hidden_size=self.model_params.hidden_state_size, 
                                     num_heads=self.model_params.num_heads, 
                                     max_sequence_length=self.model_params.sequence_length, 
                                     num_layers=self.model_params.num_layers).to(self.device)

        # initialize the dataset and dataloader
        if data_source is not None:
            self._init_data(data_source)
            self.data_source = data_source

            # set the optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_params.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

            self.save_path = f'{save_folder}/{directory}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            # define save location
            
            self.ckpt_path = f'{self.save_path}/checkpoints'
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)

            self.viz_path = f'{self.save_path}/viz'
            if not os.path.exists(self.viz_path):
                os.makedirs(self.viz_path)

        # src_mask
        self.src_mask = torch.triu(torch.ones(self.model_params.sequence_length, self.model_params.sequence_length), diagonal=1).bool().to(device)

        # init scales for animations
        self._init_scales()
        self.total_animations = 0
        self.total_real_animations = 0

        # loss functions
        self.loss_criterion = nn.MSELoss(reduction='none')
        
        
    def _init_data(self, data_source):
        all_datafiles = []
        if type(data_source) == list:
            for i in range(len(data_source)):
                with open(data_source[i], 'rb') as file:
                    all_datafiles += pickle.load(file)
        else:
            with open(data_source, 'rb') as file:
                all_datafiles += pickle.load(file)

        # split the data into train, val, test
        np.random.shuffle(all_datafiles)
        split_idx = int(len(all_datafiles)*self.train_params.train_test_split)
        train_files = all_datafiles[:split_idx]
        val_files = all_datafiles[split_idx:]

        # initialize the datasets
        self.train_dataset = VelocityTransformerDataset(self.data_params, train_files, sequence_length=self.model_params.sequence_length)
        self.val_dataset = VelocityTransformerDataset(self.data_params, val_files, sequence_length=self.model_params.sequence_length)

        print(len(self.train_dataset), len(self.val_dataset))

        # initialize the dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_params.train_batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.train_params.val_batch_size, shuffle=True, num_workers=4)

    def _init_scales(self):
        self.pose_scale = torch.tensor([1/0.25, 1, 3.14], device=self.device)
        self.targ_scale = torch.tensor([1, 1] * 2, device=self.device)
    
    def compute_loss(self, targ, out, mask):
        loss = self.loss_scales.velocity_scale * self.loss_criterion(targ, out)
        loss = torch.sum(loss*mask)/torch.sum(mask)
        return loss
    
    def train(self):
        ctr = 0
        for epoch in range(self.train_params.epochs):
            self.model.train()
            for i, (inp, targ, mask) in enumerate(self.train_loader):
                inp, targ, mask = inp.to(self.device), targ.to(self.device), mask.to(self.device)# , fsw.to(self.device)
                out = self.model(inp, src_mask = self.src_mask)
                loss = self.compute_loss(targ, out, mask)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ctr += 1
                if ctr % self.logging.print_every == 0:
                    print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
                if ctr % self.logging.eval_every == 0:
                    val_loss = self.eval()
                    self.scheduler.step(val_loss)
                    self.model.train()
                if ctr % self.logging.save_every == 0:
                    torch.save(self.model.state_dict(), f"{self.ckpt_path}/transformer_weights_{epoch}.pt")
                
    def eval(self):
        self.model.eval()
        val_loss = []
        if self.logging.animation:
            animations_ctr = 0
        with torch.no_grad():
            for i, (inp, targ, mask) in enumerate(self.val_loader):
                inp, targ, mask = inp.to(self.device), targ.to(self.device), mask.to(self.device)
                out = self.model(inp, src_mask = self.src_mask)
                loss = self.compute_loss(targ, out, mask)
                val_loss.append(loss.item())
                # add metrics here to measure performance
                
        print(f"Validation Loss: {np.mean(val_loss)}")
        return np.mean(val_loss)

    def predict(self, inp):
        self.model.eval()
        with torch.no_grad():
            inp = inp.to(self.device)
            out = self.model(inp, src_mask = self.src_mask)
            return out
    
    def load_model(self, model_path, device='cuda:0'):
        self.model.load_state_dict(torch.load(model_path))
        print('model loaded')
        self.device = device
        self.model = self.model.to(self.device)