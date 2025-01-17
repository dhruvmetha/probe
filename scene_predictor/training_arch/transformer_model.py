from data_new import TransformerDataset, RealTransformerDataset
from model import MiniTransformer
from runner_config import RunCfg
from torch.utils.data import DataLoader
from scene_predictor.evaluate_iou import get_bbox_intersections
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
import random
import sys

FFwriter = animation.FFMpegWriter

class TransformerModel:
    def __init__(self, cfg: RunCfg.transformer, data_source, save_folder, directory, device='cuda:0'):

        self.cfg = cfg

        self.model_params = cfg.model_params
        self.train_params = cfg.train_params
        self.data_params = cfg.data_params
        self.loss_scales = cfg.loss_scales
        self.logging = cfg.logging

        self.input_args = cfg.data_params.inputs.keys()
        self.input_size = sum(cfg.data_params.inputs.values())
        self.output_args = cfg.data_params.outputs
        self.output_size = sum(cfg.data_params.outputs.values())
        self.num_obstacles = cfg.data_params.obstacles

        # set the device
        self.device = device

        # initialize the model
        self.model = MiniTransformer(input_size=self.input_size, 
                                     output_size=self.output_size,
                                     num_obstacles=self.num_obstacles, 
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
        self.confidence_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.contact_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.movable_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        
    def _init_data(self, data_source):
        all_datafiles = []
        if type(data_source) == list:
            for i in range(len(data_source)):
                with open(data_source[i], 'rb') as file:
                    prev_len = len(all_datafiles)
                    all_datafiles += pickle.load(file)
                    print(data_source[i], len(all_datafiles) - prev_len)
        else:
            with open(data_source, 'rb') as file:
                all_datafiles += pickle.load(file)

        # split the data into train, val, test
        np.random.shuffle(all_datafiles)
        split_idx = int(len(all_datafiles)*self.train_params.train_test_split)
        train_files = all_datafiles[:split_idx]
        val_files = all_datafiles[split_idx:]

        # initialize the datasets
        self.train_dataset = TransformerDataset(self.data_params, train_files, sequence_length=self.model_params.sequence_length)
        self.val_dataset = TransformerDataset(self.data_params, val_files, sequence_length=self.model_params.sequence_length)

        print(len(self.train_dataset), len(self.val_dataset))

        # initialize the dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_params.train_batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.train_params.val_batch_size, shuffle=True, num_workers=4)

    def _init_scales(self):
        self.pose_scale = torch.tensor([1/0.25, 1, 3.14], device=self.device)
        targ_scale = []
        fsw_scale = []
        if 'confidence' in self.output_args:
            targ_scale.append(1)
            fsw_scale.append(1)
        
        if 'contact' in self.output_args:
            targ_scale.append(1)
            fsw_scale.append(1)
        
        if 'movable' in self.output_args:
            targ_scale.append(1)
            fsw_scale.append(1)
        
        if 'pose' in self.output_args:
            targ_scale.append(1/0.25)
            targ_scale.append(1)
            targ_scale.append(3.14)

            fsw_scale.append(1)
            fsw_scale.append(1)
            fsw_scale.append(1)

        if 'size' in self.output_args:
            targ_scale.append(1)
            targ_scale.append(1.7)
            
            fsw_scale.append(1)
            fsw_scale.append(1)

        #     self.targ_scale = torch.tensor([1, 1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 2, device=self.device)
        #     self.fsw_scale = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1] * self.num_obstacles)
        # else:
        
        self.targ_scale = torch.tensor(targ_scale * self.num_obstacles, device=self.device)
        self.fsw_scale = torch.tensor(fsw_scale * self.num_obstacles)

    def loss_func(self, targ, out, mask):
        k = 0
        confidence_mask = torch.ones((targ.shape[0], targ.shape[1], 1), device=self.device, requires_grad=False)
        targ_clone = targ.clone()
        
        # print("######")
        # object 1
        
        loss_list = []
        
        # poses = []
        for obs_idx in range(self.num_obstacles):
            obstacle_losses = []
            # print("#####", obs_idx)
            if 'confidence' in self.output_args:
                confidence_mask = (targ_clone[:, :, k+0:k+1]).float()
                loss_list.append(torch.sum(self.confidence_loss(out[:, :, k+0:k+1], targ[:, :, k+0:k+1]) * mask)/torch.sum(mask))
                k += self.output_args['confidence']

                # obstacle_losses.append(loss_list[-1].item())
                # print('confidence', loss_list[-1].item())

            if 'contact' in self.output_args:
                loss_list.append(torch.sum(self.contact_loss(out[:, :, k+0:k+1], targ[:, :, k+0:k+1]) * mask)/torch.sum(mask))
                k += self.output_args['contact']
                # obstacle_losses.append(loss_list[-1].item())
                # print('contact', loss_list[-1].item())

            if 'movable' in self.output_args:
                loss_list.append(torch.sum((self.movable_loss(out[:, :, k+0:k+1], targ[:, :, k+0:k+1]) *  confidence_mask) * mask)/torch.sum(mask))
                k += self.output_args['movable']
                # obstacle_losses.append(loss_list[-1].item())
                # print('movable', loss_list[-1].item())

            if 'pose' in self.output_args:
                # poses.append(out[:, :, k:k+2])
                loss_list.append(torch.sum((self.pos_loss(out[:, :, k:k+1], targ[:, :, k:k+1]) *  confidence_mask) * mask)/torch.sum(mask))
                k += self.output_args['pose']//3
                # obstacle_losses.append(loss_list[-1].item())
                # print('pose x', loss_list[-1].item())

                loss_list.append(torch.sum((self.pos_loss(out[:, :, k:k+1], targ[:, :, k:k+1]) * 2 *  confidence_mask) * mask)/torch.sum(mask))
                k += 1 # self.output_args['pose']//3
                # obstacle_losses.append(loss_list[-1].item())
                # print('pose y', loss_list[-1].item())

                loss_list.append(torch.sum((self.yaw_loss(out[:, :, k:k+1], targ[:, :, k:k+1]) *  confidence_mask) * mask)/torch.sum(mask))
                k += 1 # self.output_args['pose']//3
                # obstacle_losses.append(loss_list[-1].item())
                # print('yaw', loss_list[-1].item())

            if 'size' in self.output_args:
                loss_list.append(torch.sum((self.size_loss(out[:, :, k:k+1], targ[:, :, k:k+1]) *  confidence_mask) * mask)/torch.sum(mask))
                k += 1 # self.output_args['size']//2

                loss_list.append(torch.sum((self.size_loss(out[:, :, k:k+1], targ[:, :, k:k+1]) *  confidence_mask) * mask)/torch.sum(mask))
                k += 1 # self.output_args['size']//2

                # obstacle_losses.append(loss_list[-1].item())
                # print('size', loss_list[-1].item())

            # print(np.sum(obstacle_losses)) 
        # print(0.4torch.norm(poses[0][:, :, :2] - poses[1][:, :, :2], dim=-1) * mask.squeeze(0))
        # loss_list.append(torch.sum(5 * torch.relu(0.16 - torch.square(poses[0][:, :, :2] - poses[1][:, :, :2],)) * mask)/torch.sum(mask))
        # loss_list.append(torch.sum(5 * torch.relu(0.16 - torch.square(poses[0][:, :, 1:2] - poses[1][:, :, 1:2]))* mask)/torch.sum(mask))
            
        return loss_list
        
    def train(self):
        ctr = 0
        for epoch in range(self.train_params.epochs):
            self.model.train()
            for i, (inp, targ, mask, _, _, _) in enumerate(self.train_loader):
                inp, targ, mask = inp.to(self.device), targ.to(self.device), mask.to(self.device)# , fsw.to(self.device)
                
                out = self.model(inp, src_mask = self.src_mask)
                loss_list = self.loss_func(targ, out, mask)
                loss = torch.sum(torch.stack(loss_list))
                # loss1 = torch.sum(loss1*mask)/torch.sum(mask)
                # loss2 = torch.sum(loss2*mask)/torch.sum(mask)
                # loss = loss1 + loss2
                
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
            for i, (inp, targ, mask, fsw, pose, _) in enumerate(self.val_loader):
                inp, targ, mask = inp.to(self.device), targ.to(self.device), mask.to(self.device)
                
                out = self.model(inp, src_mask = self.src_mask)
                loss_list = self.loss_func(targ, out, mask)
                loss = torch.sum(torch.stack(loss_list))
                # loss1 = torch.sum(loss1*mask)/torch.sum(mask)
                # loss2 = torch.sum(loss2*mask)/torch.sum(mask)
                # loss = loss1 + loss2
                
                val_loss.append(loss.item())

                # add metrics here to measure performance

                # create animations
                if self.logging.animation and animations_ctr < 2:
                    pose = pose.to(self.device)
                    self.save_animation(pose, targ, mask, fsw, out)
                    animations_ctr += 1
        print(f"Validation Loss: {np.mean(val_loss)}")
        return np.mean(val_loss)

    def test(self, test_data_source):
        # initialize the datasets
        test_datafiles = []
        if type(test_data_source) == list:
            for i in range(len(test_data_source)):
                with open(test_data_source[i], 'rb') as file:
                    tmp_datafiles = pickle.load(file)
                    random.shuffle(tmp_datafiles)
                    test_datafiles += tmp_datafiles
        else:
            with open(test_data_source, 'rb') as file:
                test_datafiles += pickle.load(file)
        
        random.shuffle(test_datafiles)
        # test_datafiles[:500]
        # test_datafiles = glob.glob(f"{test_data_source}")
        test_dataset = TransformerDataset(self.data_params, test_datafiles[:500], sequence_length=self.model_params.sequence_length)

        # initialize the dataloaders
        test_loader = DataLoader(test_dataset, batch_size=self.train_params.test_batch_size, shuffle=True)

        record_seq_loss = {
            'contact': {
                'movable': [],
                'static': []
                },
            'movable': {
                'movable': [],
                'static': []
                },
            'pose': {
                'movable': [],
                'static': []
                },
            'size': {
                'movable': [],
                'static': []
                },
            'iou': {
                'movable': [],
                'static': []
            }
        }

        record_final_loss = {
            'contact': {
                'movable': [],
                'static': []
                },
            'movable': {
                'movable': [],
                'static': []
                },
            'pose': {
                'movable': [],
                'static': []
                },
            'size': {
                'movable': [],
                'static': []
                },
            'iou': {
                'movable': [],
                'static': []
            }
        }

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(total=len(test_dataset))
            for i, (inp, targ, mask, fsw, pose, gt_target) in enumerate(test_loader):
                
                inp, targ, mask = inp.to(self.device), targ.to(self.device), mask.to(self.device)
                
                done_idx = (~mask[0]).nonzero()[0][0]
                
                out = self.model(inp, src_mask = self.src_mask)
                
                out *= self.targ_scale
                targ *= self.targ_scale

                out_iou = out.clone().cpu().numpy()
                targ_iou = targ.clone().cpu().numpy()

                k = 0
                gt_k = 0

                contact_end_idx = 0
                for obs_idx in range(self.num_obstacles):
                    if obs_idx == 0:
                        sub_key = 'movable'
                    else:
                        sub_key = 'static'

                    contact_points = (torch.sigmoid(out[:, :done_idx, k]) > 0.8).nonzero()
                    if len(contact_points) > 0:
                        if contact_points[-1][-1] > contact_end_idx:
                            contact_end_idx = contact_points[-1][-1]
                    k += self.output_size

                k = 0                    
                for obs_idx in range(self.num_obstacles):
                    if obs_idx == 0:
                        sub_key = 'movable'
                    else:
                        sub_key = 'static'
                    
                    contact_points = (torch.sigmoid(out[:, :done_idx, k]) > 0.8).nonzero()
                    if len(contact_points) > 0:
                        contact_idx = contact_points[0][-1]
                        contact_start_idx = contact_points[0][-1]
                        # contact_end_idx = max(contact_end_idx, contact_points[-1][-1])
                        if contact_start_idx  == contact_end_idx:
                            gt_k += 9
                            k += self.output_size
                            continue
                    else:
                        gt_k += 9
                        k += self.output_size
                        continue

                    if 'confidence' in self.output_args:
                        k += self.output_args['confidence']
                    
                    if 'contact' in self.output_args:
                        # record_seq_loss.append(self.contact_loss(out[:, :, k+0:k+1], targ[:, :, k+0:k+1]))
                        # record_final_loss.append(self.contact_loss(out[:, done_idx, k+0:k+1], targ[:, done_idx, k+0:k+1]))
                        k += self.output_args['contact'] # self.output_args['contact']
                    if 'movable' in self.output_args:
                        success_mov_class = ((torch.sigmoid(out[:, :, k:k+1]) > 0.5) * 1. == targ[:, :, k:k+1]) * 1.0
                        # movable_loss = F.binary_cross_entropy_with_logits(out[:, :, k:k+1], targ[:, :, k:k+1], reduction='none')
                        record_seq_loss['movable'][sub_key].append((torch.sum(success_mov_class[0, contact_start_idx:contact_end_idx, :], dim=0)/(contact_end_idx-contact_start_idx)).cpu().numpy())
                        record_final_loss['movable'][sub_key].append(success_mov_class[0, contact_end_idx, :].cpu().numpy())
                        k += self.output_args['movable']# self.output_args['movable']
                    
                    if 'pose' in self.output_args:
                        out[:, :, k:k+3] = out[:, :, k:k+3] # * self.targ_scale[2:5]
                        targ[:, :, k:k+3] = targ[:, :, k:k+3] # * self.targ_scale[2:5]
                        pos_loss = torch.abs(out[:, :, k:k+3] - targ[:, :, k:k+3])
                        
                        # print(torch.sum(pos_loss[0, first_contact:done_idx, :], dim=0), (pos_loss[0, first_contact:done_idx, :]).shape, (done_idx-first_contact), pos_loss[0, done_idx, :].cpu().numpy())

                        # plt.plot(pos_loss[0, first_contact:done_idx, 0].cpu().numpy())
                        # plt.savefig(f'plot_{obs_idx}.png')
                        # plt.cla()

                        record_seq_loss['pose'][sub_key].append((torch.sum(pos_loss[0, contact_start_idx:contact_end_idx, :], dim=0)/(contact_end_idx-contact_start_idx)).cpu().numpy())
                        record_final_loss['pose'][sub_key].append(pos_loss[0, contact_end_idx, :].cpu().numpy())
                        k += self.output_args['pose']
                    
                    if 'size' in self.output_args:
                        out[:, :, k:k+2] = out[:, :, k:k+2] # * self.targ_scale[5:7]
                        targ[:, :, k:k+2] = targ[:, :, k:k+2] # * self.targ_scale[5:7]
                        # size_loss = F.mse_loss(out[:, :, k:k+2], targ[:, :, k:k+2], reduction='none')
                        size_loss = torch.abs(out[:, :, k:k+2] - targ[:, :, k:k+2])
                        record_seq_loss['size'][sub_key].append((torch.sum(size_loss[0, contact_start_idx:contact_end_idx, :], dim=0)/(contact_end_idx-contact_start_idx)).cpu().numpy())
                        record_final_loss['size'][sub_key].append(size_loss[0, contact_end_idx, :].cpu().numpy())
                        k += self.output_args['size']

                    
                    intersection_seq, union_seq = get_bbox_intersections(targ_iou[:, contact_start_idx:contact_end_idx, k-5:k], out_iou[:, contact_start_idx:contact_end_idx, k-5:k])

                    record_seq_loss['iou'][sub_key].append(np.mean(intersection_seq/union_seq))

                    intersection_final, union_final = get_bbox_intersections(targ_iou[:, contact_end_idx:contact_end_idx+1, k-5:k], out_iou[:, contact_end_idx:contact_end_idx+1, k-5:k])

                    record_final_loss['iou'][sub_key].append(np.mean(intersection_final/union_final))

                    gt_k += 9

                progress_bar.update(1)

                # if (i+1) == 50:
                #     break
        
        # get means
        for key in list(record_seq_loss.keys()):
            for sub_key in ['movable', 'static']:
                if len(record_seq_loss[key][sub_key]) > 0:
                    record_seq_loss[key][sub_key] = np.mean(record_seq_loss[key][sub_key], axis=0).tolist()
                    record_final_loss[key][sub_key] = np.mean(record_final_loss[key][sub_key], axis=0).tolist()
                else:
                    del record_seq_loss[key][sub_key]
                    del record_final_loss[key][sub_key]

        return { 'seq_loss': record_seq_loss, 'final_loss': record_final_loss }
    
    def test_real(self, data_source, log_folder):
        datafiles = data_source
        
        full_seq_data = []
        
        record_seq_loss_main = {
            'contact': {
                'movable': [],
                'static': []
                },
            'movable': {
                'movable': [],
                'static': []
                },
            'pose': {
                'movable': [],
                'static': []
                },
            'size': {
                'movable': [],
                'static': []
                },
            'iou': {
                'movable': [],
                'static': []
            }
        }
        
        record_final_loss = {
            'contact': {
                'movable': [],
                'static': []
                },
            'movable': {
                'movable': [],
                'static': []
                },
            'pose': {
                'movable': [],
                'static': []
                },
            'size': {
                'movable': [],
                'static': []
                },
            'iou': {
                'movable': [],
                'static': []
            }
        }
        
        # initialize the datasets
        real_dataset = RealTransformerDataset(self.data_params, datafiles, sequence_length=self.model_params.sequence_length)
        dataloader = DataLoader(real_dataset, batch_size=1, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for i, (inp, targ, mask, pose) in tqdm(enumerate(dataloader)):
                record_seq_loss = {
                    'contact': {
                        'movable': [],
                        'static': []
                        },
                    'movable': {
                        'movable': [],
                        'static': []
                        },
                    'pose': {
                        'movable': [],
                        'static': []
                        },
                    'size': {
                        'movable': [],
                        'static': []
                        },
                    'iou': {
                        'movable': [],
                        'static': []
                    },
                    'contact_window':{
                        'movable': [],
                        'static': []
                    },
                    'datafile': datafiles[i]
                }
                # record_seq_loss['datafile'] = datafiles[i]

                inp, targ, mask = inp.to(self.device), targ.to(self.device), mask.to(self.device)
                out = self.model(inp, src_mask = self.src_mask)

                done_idx = (mask[0]).nonzero()[-1][0]

                out *= self.targ_scale

                if self.logging.animation:
                    print('saving animations')
                    pose = pose.to(self.device)
                    pose *= self.pose_scale
                    self.save_real_animation(pose, targ, out, mask, log_folder)

                final_contact_idx = 0
                k = 0 
                for sub_key in ['movable', 'static']:
                    if 'confidence' in self.output_args:
                        conf_thresh = 0.6 if sub_key == 'static' else 0.6
                        confidence = (torch.sigmoid(out[0, :done_idx, k]) > conf_thresh) * 1.0
                        contact_points = confidence.nonzero()
                        if len(contact_points) > 0:
                            contact_end_idx = contact_points[-1][0]
                            print(sub_key, contact_end_idx)
                            if contact_end_idx > final_contact_idx:
                                final_contact_idx = contact_end_idx
                        k += self.output_size

                # add metrics here to measure performance
                k = 0
                targ_k = 0
                start_k = 0
                start_targ_k = 0
                for sub_key in ['movable', 'static']:
                    contact_start_idx, contact_end_idx, contact_idx = 0, 0, None
                    if 'confidence' in self.output_args:
                        conf_thresh = 0.6 if sub_key == 'static' else 0.6
                        confidence = (torch.sigmoid(out[0, :done_idx, k]) > conf_thresh) * 1.0
                        contact_points = confidence.nonzero()
                        if len(contact_points) > 0:
                            contact_start_idx = contact_points[0][0]
                            print(sub_key, contact_start_idx)
                            record_seq_loss['contact_window'][sub_key] = [contact_start_idx.item(), final_contact_idx.item()]
                            # print('contact', contact_start_idx, final_contact_idx, done_idx)
                        k += self.output_args['confidence']
                        start_k += self.output_args['confidence']

                    if 'contact' in self.output_args:
                        k += self.output_args['contact']
                        start_k += self.output_args['contact']

                    if 'movable' in self.output_args:
                        k += self.output_args['movable']
                        start_k += self.output_args['movable']

                    if 'pose' in self.output_args:
                        pos_loss = torch.abs(out[:, :, k:k+3] - targ[:, :, targ_k:targ_k+3])
                        record_final_loss['pose'][sub_key].append(pos_loss[0, final_contact_idx, :].cpu().numpy())
                        # record_seq_loss['pose'][sub_key].append(pos_loss[0, contact_start_idx:final_contact_idx, :].clone().cpu().numpy().tolist())
                        k += self.output_args['pose']
                        start_k += self.output_args['pose']
                        targ_k += self.output_args['pose']
                        
                    
                    if 'size' in self.output_args:
                        size_loss = torch.abs(out[:, :, k:k+2] - targ[:, :, targ_k:targ_k+2])
                        record_final_loss['size'][sub_key].append(size_loss[0, final_contact_idx, :].cpu().numpy())
                        # record_seq_loss['size'][sub_key].append(size_loss[0, contact_start_idx:final_contact_idx, :].clone().cpu().numpy().tolist())
                        k += self.output_args['size']
                        start_k += self.output_args['size']
                        targ_k += self.output_args['size']
                        

                    out_iou = out.clone().cpu().numpy()
                    targ_iou = targ.clone().cpu().numpy()
                    
                    if targ[0, 2, start_targ_k] > 0.:
                        intersection, union = get_bbox_intersections(targ_iou[:, contact_start_idx:final_contact_idx, start_targ_k:start_targ_k+targ_k], out_iou[:, contact_start_idx:final_contact_idx, start_k-5:start_k])
                        # print(intersection.shape, union.shape, (intersection[0, :]/union[0, :]).shape)
                        record_seq_loss['iou'][sub_key].append((intersection[0, :]/union[0, :]).tolist())
                        
                        out_iou = out.clone().cpu().numpy()
                        targ_iou = targ.clone().cpu().numpy()

                        intersection, union = get_bbox_intersections(targ_iou[:, final_contact_idx-1:final_contact_idx, start_targ_k:start_targ_k+targ_k], out_iou[:, final_contact_idx-1:final_contact_idx, start_k-5:start_k])
                        record_final_loss['iou'][sub_key].append((intersection[0, :]/union[0, :]))



                    start_targ_k += targ_k

                full_seq_data.append(record_seq_loss)
                    
        # get means
        
        for key in list(record_final_loss.keys()):
            for sub_key in ['movable', 'static']:
                if len(record_final_loss[key][sub_key]) > 0:
                    record_final_loss[key][sub_key] = np.mean(record_final_loss[key][sub_key], axis=0).tolist()
                else:
                    del record_final_loss[key][sub_key]

        return {'full_seq_data': full_seq_data, 'final_loss': record_final_loss}
                    
    def save_real_animation(self, pose, targ, out, mask, log_folder):
        # save animations
        patches = []
        done_idx = (mask[0]).nonzero()[-1][0]
        contact_1_first, contact_1_last, contact_2_first, contact_2_last = 1500, 0, 1500, 0
        contact_1 = (torch.sigmoid(out[0, :done_idx, 0])>0.6).nonzero()
        if len(contact_1) > 0:
            contact_1_first = contact_1[0, 0]
            contact_1_last = contact_1[-1, 0]
        contact_2 = (torch.sigmoid(out[0, :done_idx, 8])>0.6).nonzero()
        if len(contact_2) > 0:
            contact_2_first = contact_2[0, 0]
            contact_2_last = contact_2[-1, 0]

        first_contact = min(contact_1_first, contact_2_first)
        last_contact = max(contact_1_last, contact_2_last)

        for step in range(pose.shape[1]):
            pose = pose.to('cpu')
            targ = targ.to('cpu')
            out = out.to('cpu')
            mask = mask.to('cpu')

            if step < first_contact:
                mod_out = torch.zeros_like(out[0, step, :])
            else:
                step_idx_1 = step if step < last_contact else last_contact
                step_idx_2 = step if step < last_contact else last_contact

                if step < contact_1_first:
                    _first = torch.zeros_like(out[0, step_idx_1, :8])
                else:
                    _first = out[0, step_idx_1, :8]
                if step < contact_2_first:
                    _second = torch.zeros_like(out[0, step_idx_2, 8:])
                else:
                    _second = out[0, step_idx_2, 8:]
                
                mod_out = torch.cat([_first, _second], dim=-1)
            
            if mask[0, step, 0]:
                patch_set = get_real_visualization(self.num_obstacles, pose[0, step, :], targ[0, step, :], mod_out)
                patches.append(patch_set)
            else:
                break

        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
        ax.axis('off')
        ax.set(xlim=(-1.0, 4.0), ylim=(-1, 1), aspect='auto')
        last_patch = []

        def animate(frame):
            if len(last_patch) != 0:
                for i in last_patch:
                    try:
                        i.remove()
                    except:
                        pass
                last_patch.clear()
        
            for patch in frame:
                if patch is not None:
                    ax.add_patch(patch)
                    last_patch.append(patch)

        # print(len(patches))
        anim = animation.FuncAnimation(fig, animate, frames=patches, interval=10, repeat=False)
        
        anim.save(f"{log_folder}/plot_{self.total_real_animations}.mp4", writer = FFwriter(10))
        plt.close()
        # with open(f'{log_folder}/real_plot_{self.total_real_animations}.pkl', 'wb') as f:
        #         pickle.dump(patches, f)
        self.total_real_animations += 1

    def save_animation(self, pose, targ, mask, fsw, out):
        # save animations
        patches = []

        for step in range(pose.shape[1]):
            # print(step)
            if mask[0, step, 0]:
                patch_set = get_visualization(self.output_args, self.output_size, self.num_obstacles,  0, pose[:, step, :]*self.pose_scale, targ[:, step, :]*self.targ_scale, pose[:, step, :]* self.pose_scale, out[:, step, :]*self.targ_scale, fsw[:, step, :]*self.fsw_scale)
                patches.append(patch_set)
            else:
                break
        print(len(patches))
        with open(f'{self.viz_path}/plot_{self.total_animations}.pkl', 'wb') as f:
                pickle.dump(patches, f)
        self.total_animations += 1

    def load_model(self, model_path, device='cuda:0'):
        self.model.load_state_dict(torch.load(model_path))
        print('model loaded')
        self.device = device
        self.model = self.model.to(self.device)

    def confidence_loss(self, out, targ):
        l = self.loss_scales.confidence_scale * self.confidence_criterion(out, targ)
        return l

    def contact_loss(self, out, targ):
        l = self.loss_scales.contact_scale * self.contact_criterion(out, targ)
        return l

    def movable_loss(self, out, targ):
        l = self.loss_scales.movable_scale * self.movable_criterion(out, targ)
        # print(l)
        return l
    
    def pos_loss(self, out, targ):
        l = self.loss_scales.pos_scale * F.mse_loss(out, targ, reduction='none')
        return l
    
    def yaw_loss(self, out, targ):
        l = self.loss_scales.yaw_scale * F.mse_loss(out, targ, reduction='none')
        return l
    
    def size_loss(self, out, targ):
        return self.loss_scales.size_scale * F.mse_loss(out, targ, reduction='none')
