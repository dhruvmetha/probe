from data import TransformerDataset
from model import MiniTransformer
from torch.utils.data import DataLoader
from params_proto import PrefixProto
from visualization import get_visualization
import glob
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

class TransformerArgs(PrefixProto):
    window_size = 250
    sequence_length = 750
    hidden_state_size = 1024
    embed_size = 512
    num_heads = 2
    num_layers = 2
    alg = 'transformer'
    eval_every = 250
    test_every = eval_every
    save_every = 750
    print_every = 50
    epochs = 100
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 1
    learning_rate = 1e-4
    dropout = 0.
    train_test_split = 0.95
    estimate_pose = False
    input_size = 39
    output_size = 7
    boxes = 2
    animation = True

    save_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/'

    contact_scale = 1/3
    movable_scale = 1/2
    pos_scale = 2
    yaw_scale = 2 * 10
    size_scale = 2

class TransformerModel:
    def __init__(self, data_source, device='cuda:0'):

        # set the device
        self.device = device

        # initialize the model
        self.model = MiniTransformer(input_size=TransformerArgs.input_size, 
                                     output_size=TransformerArgs.output_size * TransformerArgs.boxes, 
                                     embed_size=TransformerArgs.embed_size, 
                                     hidden_size=TransformerArgs.hidden_state_size, 
                                     num_heads=TransformerArgs.num_heads, 
                                     max_sequence_length=TransformerArgs.sequence_length, 
                                     num_layers=TransformerArgs.num_layers).to(self.device)

        # initialize the dataset and dataloader
        self._init_data(data_source)

        # set the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=TransformerArgs.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        

        # src_mask
        self.src_mask = torch.triu(torch.ones(TransformerArgs.sequence_length-1, TransformerArgs.sequence_length-1), diagonal=1).bool().to(device)

        # init scales for animations
        self._init_scales()
        self.total_animations = 0

        # define save location
        self.save_path = f'{TransformerArgs.save_path}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        
    def _init_data(self, data_source):

        all_datafiles = glob.glob(f"{data_source}")

        # split the data into train, val, test
        np.random.shuffle(all_datafiles)
        split_idx = int(len(all_datafiles)*TransformerArgs.train_test_split)
        train_files = all_datafiles[:split_idx]
        val_files = all_datafiles[split_idx:]

        # initialize the datasets
        self.train_dataset = TransformerDataset(train_files, sequence_length=TransformerArgs.sequence_length)
        self.val_dataset = TransformerDataset(val_files, sequence_length=TransformerArgs.sequence_length)

        # initialize the dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=TransformerArgs.train_batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=TransformerArgs.val_batch_size, shuffle=True, num_workers=4)

    def _init_scales(self):
        self.pose_scale = torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=self.device)
        self.targ_scale = torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=self.device)
        self.fsw_scale = torch.tensor([1, 1, 1, 1, 1, 1], device=self.device)

    def loss_fn(self, out, targ):
        k = 0

        loss1 = F.binary_cross_entropy(torch.sigmoid(out[:, :, k:k+1]), targ[:, :, k:k+1], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+7:k+8]), targ[:, :, k+7:k+8], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+14:k+15]), targ[:, :, k+14:k+15], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, 21:22]), targ[:, :, 21:22], reduction='none')

        loss2 = F.binary_cross_entropy(torch.sigmoid(out[:, :, k+1:k+2]), targ[:, :, k+1:k+2], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+8:k+9]), targ[:, :, k+8:k+9], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+15:k+16]), targ[:, :, k+15:k+16], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, 22:23]), targ[:, :, 22:23], reduction='none')

        loss3 = F.mse_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4], reduction='none') + F.mse_loss(out[:, :, k+9:k+11], targ[:, :, k+9:k+11], reduction='none') + F.mse_loss(out[:, :, k+16:k+18], targ[:, :, k+16:k+18], reduction='none') # + F.mse_loss(out[:, :, 23:25], targ[:, :, 23:25], reduction='none')
        loss3 = torch.sum(loss3, dim=-1).unsqueeze(-1)

        loss4 = F.mse_loss(out[:, :, k+4:k+7], targ[:, :, k+4:k+7], reduction='none') + F.mse_loss(out[:, :, k+11:k+14], targ[:, :, k+11:k+14], reduction='none') + F.mse_loss(out[:, :, k+18:k+21], targ[:, :, k+18:k+21], reduction='none') # + F.mse_loss(out[:, :, 25:], targ[:, :, 25:], reduction='none')
        loss4 = torch.sum(loss4, dim=-1).unsqueeze(-1)

        return loss1, loss2, loss3, loss4
    
    def loss_func(self, targ, out):
                
        k = 0
        losses = []
        for i in range(TransformerArgs.boxes):
            # object 1
            contact_loss = self.contact_loss(out[:, :, k+0:k+1], targ[:, :, k+0:k+1])
            movable_loss = self.movable_loss(out[:, :, k+1:k+2], targ[:, :, k+1:k+2])
            pos_loss = self.pos_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4])
            yaw_loss = self.yaw_loss(out[:, :, k+4:k+5], targ[:, :, k+4:k+5])
            size_loss = self.size_loss(out[:, :, k+5:k+7], targ[:, :, k+5:k+7])
            loss = contact_loss + movable_loss + pos_loss + yaw_loss + size_loss
            losses.append(loss)
            k += TransformerArgs.output_size
        
        return losses

    def train(self, epochs=TransformerArgs.epochs):
        ctr = 0
        for epoch in range(epochs):
            self.model.train()
            for i, (inp, targ, mask, fsw, _) in enumerate(self.train_loader):
                inp, targ, mask = inp.to(self.device), targ.to(self.device), mask.to(self.device)# , fsw.to(self.device)
                
                out = self.model(inp, src_mask = self.src_mask)
                losses = self.loss_func(out, targ)
                loss = None
                for l in losses:
                    if loss is None:
                        loss = torch.sum(l*mask)/torch.sum(mask)
                    else:
                        loss += torch.sum(l*mask)/torch.sum(mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ctr += 1
                if ctr % TransformerArgs.print_every == 0:
                    print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
                if ctr % TransformerArgs.eval_every == 0:
                    val_loss = self.eval()
                    self.scheduler.step(val_loss)
                    self.model.train()
                if ctr % TransformerArgs.save_every == 0:
                    torch.save(self.model.state_dict(), f"checkpoints/transformer_weights_{epoch}.pt")
                
    def eval(self):
        self.model.eval()
        val_loss = []
        if TransformerArgs.animation:
            animations_ctr = 0
        with torch.no_grad():
            for i, (inp, targ, mask, fsw, pose) in enumerate(self.val_loader):
                inp, targ, mask = inp.to(self.device), targ.to(self.device), mask.to(self.device)
                
                out = self.model(inp, src_mask = self.src_mask)
                losses = self.loss_func(out, targ)
                loss = None
                for l in losses:
                    if loss is None:
                        loss = torch.sum(l*mask)/torch.sum(mask)
                    else:
                        loss += torch.sum(l*mask)/torch.sum(mask)
                val_loss.append(loss.item())

                # add metrics here to measure performance

                # create animations
                if TransformerArgs.animation and animations_ctr < 2:
                    pose, fsw = pose.to(self.device), fsw.to(self.device)
                    self.save_animation(pose, targ, mask, fsw, out)
                    animations_ctr += 1
                
        print(f"Validation Loss: {np.mean(val_loss)}")


        if TransformerArgs.animation:
            # saving animations
            print("Saving animations", len(animations))
            for i in range(len(animations)):
                with open(f'plot_{self.total_animations}.pkl', 'wb') as file:
                    pickle.dump(animations, file)
                self.total_animations += 1
        
        return np.mean(val_loss)

    def test(self, test_data_source):
         # initialize the datasets
        test_datafiles = glob.glob(f"{test_data_source}")
        test_dataset = TransformerDataset(test_datafiles, sequence_length=TransformerArgs.sequence_length)

        # initialize the dataloaders
        test_loader = DataLoader(test_dataset, batch_size=TransformerArgs.test_batch_size, shuffle=True, num_workers=4)

        record_losses = {
            'contact': [],
            'movable': [],
            'pos': [],
            'yaw': [],
            'size': []
        }

        self.model.eval()
        with torch.no_grad():
            for i, (inp, targ, mask, fsw, pose) in enumerate(test_loader):
                inp, targ, mask = inp.to(self.device), targ.to(self.device), mask.to(self.device)
                
                out = self.model(inp, src_mask = self.src_mask)
                k = 0
                contact_loss = self.contact_loss(out[:, :, k+0:k+1], targ[:, :, k+0:k+1])
                movable_loss = self.movable_loss(out[:, :, k+1:k+2], targ[:, :, k+1:k+2])
                pos_loss = self.pos_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4])
                yaw_loss = self.yaw_loss(out[:, :, k+4:k+5], targ[:, :, k+4:k+5])
                size_loss = self.size_loss(out[:, :, k+5:k+7], targ[:, :, k+5:k+7])
                    
                record_losses['contact'].append(torch.sum(contact_loss*mask)/torch.sum(mask))
                record_losses['movable'].append(torch.sum(movable_loss*mask)/torch.sum(mask))
                record_losses['pos'].append(torch.sum(pos_loss*mask)/torch.sum(mask))
                record_losses['yaw'].append(torch.sum(yaw_loss*mask)/torch.sum(mask))
                record_losses['size'].append(torch.sum(size_loss*mask)/torch.sum(mask))

                # create animations
                # pose, fsw = pose.to(self.device), fsw.to(self.device)
                # self.save_animation(pose, targ, mask, fsw, out)

    def save_animation(self, pose, targ, mask, fsw, out):
        # save animations
        patches = []
        for step in range(pose.shape[1]):
            if mask[0, step, 0]:
                patch_set = get_visualization(0, pose[:, step, :]*self.pose_scale, targ[:, step, :]*self.targ_scale, pose[:, step, :]* self.pose_scale, out[:, step, :]*self.targ_scale, fsw[:, step, :]*self.fsw_scale, estimate_pose=TransformerArgs.estimate_pose)
                patches.append(patch_set)
            else:
                break
        
        with open(f'{self.save_path}/plot_{self.total_animations}.pkl', 'wb') as f:
                pickle.dump(patches, f)
        self.total_animations += 1

    def load_model(self, model_path, device='cuda:0'):
        self.model.load_state_dict(torch.load(model_path))
        print('model loaded')
        self.device = device
        self.model = self.model.to(self.device)

    def contact_loss(self, out, targ):
        return TransformerArgs.contact_scale * F.binary_cross_entropy(torch.sigmoid(out), targ, reduction='none')

    def movable_loss(self, out, targ):
        return TransformerArgs.movable_scale * F.binary_cross_entropy(torch.sigmoid(out), targ, reduction='none')

    def pos_loss(self, out, targ):
        return TransformerArgs.pos_scale * F.mse_loss(out, targ, reduction='none')

    def yaw_loss(self, out, targ):
        return TransformerArgs.yaw_scale * F.mse_loss(out, targ, reduction='none')
    
    def size_loss(self, out, targ):
        return TransformerArgs.size_scale * F.mse_loss(out, targ, reduction='none')