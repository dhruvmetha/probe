from data import PoseEstimatorDataset, SceneEstimatorDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
from torch import optim
from model import MiniTransformer
from tqdm import tqdm
from glob import glob
from visualization import get_visualization
from iou import get_bbox_intersections
import torch
import pickle
import random
import numpy as np
import wandb
import time



class SceneEstimator:
    def __init__(self, input_size, causal=False, num_layers=2, num_heads=2, device='cuda:0', embed_size=512, hidden_size=2048, start=10):
        
        self.optimizer = None
        self.scheduler = None
        self.sequence_length = 1500
        self.device = device
        self.start = start

        print("start###########", self.start)

        self.model = MiniTransformer(input_size=input_size, output_size=24, embed_size=embed_size, hidden_size=hidden_size, num_heads=num_heads, max_sequence_length=self.sequence_length, num_layers=num_layers, estimate_pose=False)

        if causal:
            self.src_mask =  torch.triu(torch.ones(self.sequence_length-self.start, self.sequence_length-self.start), diagonal=1).bool().to(self.device)
        else:
            self.src_mask = None

        self.saved_models = 0
        self.saved_videos = 0

        self.confidence_scale = 1
        self.contact_scale = 1/3
        self.movable_scale = 1/2
        self.pose_scale = 2
        self.yaw_scale = 2 * 10
        self.size_scale = 2

        self.ious = {
            'fixed': [],
            'movable': [],
        }


    def loss_fn(self, out, targ, pose=None):
        k = 1
        j = 2

        # confidence binary
        loss0 = F.binary_cross_entropy(torch.sigmoid(out[:, :, k-1:k]), targ[:, :, k-1:k], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, j+7-1:j+7]), targ[:, :, j+7-1:j+7], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+13:k+14]), targ[:, :, k+13:k+14], reduction='none')

        # contact binary
        loss1 = F.binary_cross_entropy(torch.sigmoid(out[:, :, k:k+1]), targ[:, :, k:k+1], reduction='none') * targ[:, :, k-1:k] + F.binary_cross_entropy(torch.sigmoid(out[:, :, j+7:j+8]), targ[:, :, j+7:j+8], reduction='none') * targ[:, :, j+7-1:j+7] # + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+14:k+15]), targ[:, :, k+14:k+15], reduction='none') 
        loss1 = torch.sum(loss1, dim=-1).unsqueeze(-1)

        # # movable or not movable
        loss2 = F.binary_cross_entropy(torch.sigmoid(out[:, :, k+1:k+2]), targ[:, :, k+1:k+2], reduction='none') * targ[:, :, k-1:k] + F.binary_cross_entropy(torch.sigmoid(out[:, :, j+8:j+9]), targ[:, :, j+8:j+9], reduction='none') * targ[:, :, j+7-1:j+7]  # + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+15:k+16]), targ[:, :, k+15:k+16], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, 22:23]), targ[:, :, 22:23], reduction='none')
        # print(loss2.shape)

        # pose x, y
        loss3 = F.mse_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4], reduction='none') * targ[:, :, k-1:k]  + F.mse_loss(out[:, :, j+9:j+11], targ[:, :, j+9:j+11], reduction='none') * targ[:, :, j+7-1:j+7]  # + F.mse_loss(out[:, :, k+16:k+18], targ[:, :, k+16:k+18], reduction='none') # + F.mse_loss(out[:, :, 23:25], targ[:, :, 23:25], reduction='none')
        # loss3 = torch.sum(loss3, dim=-1).unsqueeze(-1)

        # # yaw
        loss4 = F.mse_loss(out[:, :, k+4:k+5], targ[:, :, k+4:k+5], reduction='none') * targ[:, :, k-1:k] + F.mse_loss(out[:, :, j+11:j+12], targ[:, :, j+11:j+12], reduction='none') * targ[:, :, j+7-1:j+7] # + F.mse_loss(out[:, :, j+18:j+19], targ[:, :, j+18:j+19], reduction='none')
        loss4 = torch.sum(loss4, dim=-1).unsqueeze(-1)

        # # width, height
        loss5 = F.mse_loss(out[:, :, k+5:k+7], targ[:, :, k+5:k+7], reduction='none') * targ[:, :, k-1:k] + F.mse_loss(out[:, :, j+12:j+14], targ[:, :, j+12:j+14], reduction='none') * targ[:, :, j+7-1:j+7] # + F.mse_loss(out[:, :, k+19:k+21], targ[:, :, k+19:k+21], reduction='none')
        loss5 = torch.sum(loss5, dim=-1).unsqueeze(-1)

        ## add physical constraints 
        # x, y, theta = pose
        # loss6 = 

        return self.confidence_scale * loss0,  self.contact_scale * loss1, self.movable_scale * loss2, self.pose_scale * loss3, self.yaw_scale * loss4, self.size_scale * loss5

    def alternate_loss_fn(self, out, targ):
        k = 0

        loss_object_1 = self.contact_scale * F.binary_cross_entropy(torch.sigmoid(out[:, :, k:k+1]), targ[:, :, k:k+1], reduction='none') + self.movable_scale * F.binary_cross_entropy(torch.sigmoid(out[:, :, k+1:k+2]), targ[:, :, k+1:k+2], reduction='none') 
        loss_object_1 += (self.pose_scale * torch.sum(F.mse_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4], reduction='none'), dim=-1).unsqueeze(-1))
        loss_object_1 += (self.yaw_scale * torch.sum(F.mse_loss(out[:, :, k+4:k+5], targ[:, :, k+4:k+5], reduction='none'), dim=-1).unsqueeze(-1))
        loss_object_1 += (self.size_scale * torch.sum(F.mse_loss(out[:, :, k+5:k+7], targ[:, :, k+5:k+7], reduction='none'), dim=-1).unsqueeze(-1))

        loss_object_2 = self.contact_scale * F.binary_cross_entropy(torch.sigmoid(out[:, :, k+7:k+8]), targ[:, :, k+7:k+8], reduction='none') + self.movable_scale * F.binary_cross_entropy(torch.sigmoid(out[:, :, k+8:k+9]), targ[:, :, k+8:k+9], reduction='none')
        loss_object_2 += (self.pose_scale * torch.sum(F.mse_loss(out[:, :, k+9:k+11], targ[:, :, k+9:k+11], reduction='none'), dim=-1).unsqueeze(-1))
        loss_object_2 += (self.yaw_scale * torch.sum(F.mse_loss(out[:, :, k+11:k+12], targ[:, :, k+11:k+12], reduction='none'), dim=-1).unsqueeze(-1))
        loss_object_2 += (self.size_scale * torch.sum(F.mse_loss(out[:, :, k+12:k+14], targ[:, :, k+12:k+14], reduction='none'), dim=-1).unsqueeze(-1))


        loss_object_3 = self.contact_scale * F.binary_cross_entropy(torch.sigmoid(out[:, :, k+14:k+15]), targ[:, :, k+14:k+15], reduction='none') + self.movable_scale * F.binary_cross_entropy(torch.sigmoid(out[:, :, k+15:k+16]), targ[:, :, k+15:k+16], reduction='none')
        loss_object_3 += (self.pose_scale * torch.sum(F.mse_loss(out[:, :, k+16:k+18], targ[:, :, k+16:k+18], reduction='none'), dim=-1).unsqueeze(-1))
        loss_object_3 += (self.yaw_scale * torch.sum(F.mse_loss(out[:, :, k+18:k+19], targ[:, :, k+18:k+19], reduction='none'), dim=-1).unsqueeze(-1))
        loss_object_3 += (self.size_scale * torch.sum(F.mse_loss(out[:, :, k+19:k+21], targ[:, :, k+19:k+21], reduction='none'), dim=-1).unsqueeze(-1))

        return loss_object_1, loss_object_2, loss_object_3

    def alternate_loss_fn2(self, out, targ):
        k = 1 # 1
        loss_object_1 = (self.contact_scale * (F.binary_cross_entropy(torch.sigmoid(out[:, :, k:k+1]), targ[:, :, k:k+1], reduction='none'))  + self.movable_scale * F.binary_cross_entropy(torch.sigmoid(out[:, :, k+1:k+2]), targ[:, :, k+1:k+2], reduction='none') + (self.pose_scale * torch.sum(F.mse_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4], reduction='none'), dim=-1).unsqueeze(-1)) + (self.yaw_scale * torch.sum(F.mse_loss(out[:, :, k+4:k+5], targ[:, :, k+4:k+5]), dim=-1).unsqueeze(-1)) + (self.size_scale * torch.sum(F.mse_loss(out[:, :, k+5:k+7], targ[:, :, k+5:k+7], reduction='none'), dim=-1).unsqueeze(-1))) * targ[:, :, k-1:k] + self.confidence_scale * (F.binary_cross_entropy(torch.sigmoid(out[:, :, k-1:k]), targ[:, :, k-1:k]))

        k += 8 # 9

        loss_object_2 = (self.contact_scale * (F.binary_cross_entropy(torch.sigmoid(out[:, :, k:k+1]), targ[:, :, k:k+1], reduction='none'))  + self.movable_scale * F.binary_cross_entropy(torch.sigmoid(out[:, :, k+1:k+2]), targ[:, :, k+1:k+2], reduction='none') + (self.pose_scale * torch.sum(F.mse_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4], reduction='none'), dim=-1).unsqueeze(-1)) + (self.yaw_scale * torch.sum(F.mse_loss(out[:, :, k+4:k+5], targ[:, :, k+4:k+5]), dim=-1).unsqueeze(-1)) + (self.size_scale * torch.sum(F.mse_loss(out[:, :, k+5:k+7], targ[:, :, k+5:k+7], reduction='none'), dim=-1).unsqueeze(-1))) * targ[:, :, k-1:k] + self.confidence_scale * (F.binary_cross_entropy(torch.sigmoid(out[:, :, k-1:k]), targ[:, :, k-1:k]))


        # phys_loss = F.relu(targ[:, :, 0:1] * targ[:, :, 8:9] *  (0.1 - torch.sqrt(((out[:, :, 3] - out[:, :, 11]) * 4) ** 2 + ((out[:, :, 4] - out[:, :, 12])) ** 2)).unsqueeze(-1))

        # phys_loss = targ[:, :, 0:1] * targ[:, :, 8:9] *  ((0.1) - torch.norm(out[:, :, 3:5] - out[:, :, 11:13], dim=-1).unsqueeze(-1))

        k += 8 # 17

        loss_object_3 = (self.contact_scale * (F.binary_cross_entropy(torch.sigmoid(out[:, :, k:k+1]), targ[:, :, k:k+1], reduction='none'))  + self.movable_scale * F.binary_cross_entropy(torch.sigmoid(out[:, :, k+1:k+2]), targ[:, :, k+1:k+2], reduction='none') + (self.pose_scale * torch.sum(F.mse_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4], reduction='none'), dim=-1).unsqueeze(-1)) + (self.yaw_scale * torch.sum(F.mse_loss(out[:, :, k+4:k+5], targ[:, :, k+4:k+5]), dim=-1).unsqueeze(-1)) + (self.size_scale * torch.sum(F.mse_loss(out[:, :, k+5:k+7], targ[:, :, k+5:k+7], reduction='none'), dim=-1).unsqueeze(-1))) * targ[:, :, k-1:k] + self.confidence_scale * (F.binary_cross_entropy(torch.sigmoid(out[:, :, k-1:k]), targ[:, :, k-1:k]))

        return loss_object_1, loss_object_2, loss_object_3

    def train(self, dl, val_dl, save_folder, print_every=50, eval_every=250, save_every=500):
        train_loss, val_loss, val_ctr = 0, 0, 0
        train_loss1, train_loss2, train_loss3, train_loss4, train_loss5 = 0, 0, 0, 0, 0
        pbar = tqdm(total=len(dl))
        for i, (inp, targ, mask, _, _) in enumerate(dl):
            self.model.train()
            inp = inp.to(self.device)
            targ = targ.to(self.device)
            mask = mask.to(self.device)
            out = self.model(inp, src_mask=self.src_mask)

            # loss1, loss2, loss3, loss4, loss5 = self.loss_fn(out, targ)
            # loss1 = torch.sum(loss1*mask)/torch.sum(mask)
            # loss2 = torch.sum(loss2*mask)/torch.sum(mask)
            # loss3 = torch.sum(loss3*mask)/torch.sum(mask)
            # loss4 = torch.sum(loss4*mask)/torch.sum(mask)
            # loss5 = torch.sum(loss5*mask)/torch.sum(mask)
            # loss = loss1 * self.contact_scale + loss2 * self.movable_scale + loss3 * self.pose_scale + loss4 * self.yaw_scale + loss5 * self.size_scale

            loss1, loss2, loss3 = self.alternate_loss_fn2(out, targ)
            loss1 = torch.sum(loss1*mask)/torch.sum(mask)
            loss2 = torch.sum(loss2*mask)/torch.sum(mask)
            loss3 = torch.sum(loss3*mask)/torch.sum(mask)
            loss = loss1 + 1.0 * loss2 + 0.0 * loss3

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # train_loss1 += self.contact_scale * loss1.item()
            # train_loss2 += self.movable_scale * loss2.item()
            # train_loss3 += self.pose_scale * loss3.item()
            # train_loss4 += self.yaw_scale * loss4.item()
            # train_loss5 += self.size_scale * loss5.item()

            train_loss1 += loss1.item()
            train_loss2 += 1 * loss2.item()
            train_loss3 += 0 * loss3.item()

            if (i+1) % print_every == 0:
                print(f'step {i+1}:', train_loss/print_every)
                wandb.log({
                    'train_loss': train_loss/print_every,
                    # 'contact loss': train_loss1/print_every,
                    # 'movable loss': train_loss2/print_every,
                    # 'pose loss': train_loss3/print_every,
                    # 'yaw loss': train_loss4/print_every,
                    # 'size loss': train_loss5/print_every,
                    # 'loss class': train_loss1/print_every,
                    'loss_object_1': train_loss1/print_every,
                    'loss_object_2': train_loss2/print_every,
                    'loss_object_3': train_loss3/print_every,
                })
                print(f'loss_object_1: {train_loss1/print_every}')
                print(f'loss_object_2: {train_loss2/print_every}')
                print(f'loss_object_3: {train_loss3/print_every}')
                train_loss, train_loss1, train_loss2, train_loss3, train_loss4, train_loss5 = 0, 0, 0, 0, 0, 0

                # print(f'loss_object_3: {train_loss3/print_every}')
            
            if (i+1) % save_every == 0:
                save_path = f'{save_folder}/checkpoints'
                PATH = f'{save_path}/model_{self.saved_models}.pt'
                torch.save(self.model.state_dict(), PATH)
                print(f'{time.time()} model saved at {PATH}')
                self.saved_models += 1

            pbar.update(1)

            if (i+1) % eval_every == 0:
                new_val_loss = self.validate(val_dl, save_folder)
                val_loss += new_val_loss
                val_ctr += 1
                wandb.log({
                    'val loss': new_val_loss,
                })
                self.model.train()

        pbar.close()
        return val_loss/val_ctr
    
    def evaluation_metrics(self, out, targ, mask):
        pass

    def validate(self, dl, save_folder):
        vis_path = f'{save_folder}/viz'
        self.model.eval()
        with torch.inference_mode():
            val_loss = 0
            pbar = tqdm(total=len(dl))
            val_loss_0, val_loss_1, val_loss_2, val_loss_3, val_loss_4, val_loss_5 = 0, 0, 0, 0, 0, 0
            for i, (inp, targ, mask, fsw, pose) in enumerate(dl):
                inp = inp.to(self.device)
                targ = targ.to(self.device)
                mask = mask.to(self.device)
                out = self.model(inp, src_mask=self.src_mask)
                pose = pose.to(self.device)

                # loss1, loss2, loss3, loss4, loss5 = self.loss_fn(out, targ)
                # loss1 = torch.sum(loss1*mask)/torch.sum(mask)
                # loss2 = torch.sum(loss2*mask)/torch.sum(mask)
                # loss3 = torch.sum(loss3*mask)/torch.sum(mask)
                # loss4 = torch.sum(loss4*mask)/torch.sum(mask)
                # loss5 = torch.sum(loss5*mask)/torch.sum(mask)
                # loss = loss1 * self.contact_scale + loss2 * self.movable_scale + loss3 * self.pose_scale + loss4 * self.yaw_scale + loss5 * self.size_scale

                loss0, loss1, loss2, loss3, loss4, loss5 = self.loss_fn(out, targ)
                loss0 = torch.sum(loss0*mask)/torch.sum(mask)
                loss1 = torch.sum(loss1*mask)/torch.sum(mask)
                loss2 = torch.sum(loss2*mask)/torch.sum(mask)
                loss3 = torch.sum(loss3*mask)/torch.sum(mask)
                loss4 = torch.sum(loss4*mask)/torch.sum(mask)
                loss5 = torch.sum(loss5*mask)/torch.sum(mask)
                loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5

                val_loss += loss.item()
                val_loss_0 += loss0.item()
                val_loss_1 += loss1.item()
                val_loss_2 += loss2.item()
                val_loss_3 += loss3.item()
                val_loss_4 += loss4.item()
                val_loss_5 += loss5.item()


                pbar.update(1)

                if i < 5:
                    self.save_visualization(inp, targ, out, fsw, pose, mask, vis_path)
                
                # if i < 10:
                #     self.compute_iou(inp, out, targ, mask)
                    
            
            # print()
            # print(f'movable iou loss', np.mean(self.ious['movable']))
            # print(f'fixed iou loss', np.mean(self.ious['fixed']))
            print(f'validation loss: {val_loss/(i+1)}')

            wandb.log({
                'val_loss': val_loss/(i+1),
                'confidence loss': val_loss_0/(i+1),
                'contact loss': val_loss_1/(i+1),
                'movable loss': val_loss_2/(i+1),
                'pose loss': val_loss_3/(i+1),
                'yaw loss': val_loss_4/(i+1),
                'size loss': val_loss_5/(i+1),

            })

            print(f'confidence loss: {val_loss_0/(i+1)}')
            print(f'contact loss: {val_loss_1/(i+1)}')
            print(f'movable loss: {val_loss_2/(i+1)}')
            print(f'pose loss: {val_loss_3/(i+1)}')
            print(f'yaw loss: {val_loss_4/(i+1)}')
            print(f'size loss: {val_loss_5/(i+1)}')

        self.scheduler.step(val_loss/(i+1))
        pbar.close()
        return val_loss/(i+1)
    
    def compute_iou(self, inp, out, targ, mask):
        boxes = (torch.cat([targ[:, :, 3:8] , targ[:, :, 11:16]], dim=-1) * torch.tensor([1/0.25, 1, 3.14, 1, 1.7]*2, device=self.device)).cpu().numpy()
        pred_boxes = (torch.cat([out[:, :, 3:8] , out[:, :, 11:16]], dim=-1) * torch.tensor([1/0.25, 1, 3.14, 1, 1.7]*2, device=self.device)).cpu().numpy()

        inter_dict = {
            'movable': [],
            'fixed': [],
        }
        unions_dict = {
            'movable': [],
            'fixed': [],
        }

        for step in range(inp.shape[1]):
            if mask[0, step, 0]:
                if targ[0, step, 3] > 0:
                    intersections, unions = get_bbox_intersections(boxes[0:1, step:step+1, :5], pred_boxes[0:1, step:step+1, :5])
                    inter_dict['movable'].append(intersections[0][0][0])
                    unions_dict['movable'].append(unions[0][0][0])

                if targ[0, step, 11] > 0:
                    intersections, unions = get_bbox_intersections(boxes[0:1, step:step+1, 5:10], pred_boxes[0:1, step:step+1, 5:10])
                    inter_dict['fixed'].append(intersections[0][0][0])
                    unions_dict['fixed'].append(unions[0][0][0])
        
        if len(inter_dict['movable']) > 0:
            inter_dict['movable'] = np.array(inter_dict['movable'])
            unions_dict['movable'] = np.array(unions_dict['movable'])
            ious = inter_dict['movable'] / unions_dict['movable']
            self.ious['movable'].extend(ious.tolist())
            
        if len(inter_dict['fixed']) > 0:
            inter_dict['fixed'] = np.array(inter_dict['fixed'])
            unions_dict['fixed'] = np.array(unions_dict['fixed'])
            ious = inter_dict['fixed'] / unions_dict['fixed']
            self.ious['fixed'].extend(ious.tolist())
        

    def save_visualization(self, inp, targ, out, fsw, pose, mask, save_path):
        k = 0
        patches = []
        for step in range(inp.shape[1]):
            if mask[0, step, 0]:
                patch_set = get_visualization(0, pose[:, step, :]*torch.tensor([1/0.25, 1, 3.14], device=self.device), targ[:, step, :]*torch.tensor([1, 1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3, device=self.device), pose[:, step, :]* torch.tensor([1/0.25, 1, 3.14], device=self.device), out[:, step, :]*torch.tensor([1, 1, 1, 1/0.25, 1, 3.14, 1, 1.7] * 3, device=self.device), fsw[:, step, :].squeeze(1), estimate_pose=False)
                patches.append(patch_set)
        
        with open(f'{save_path}/plot_{self.saved_videos}.pkl', 'wb') as f:
                pickle.dump(patches, f)
        self.saved_videos += 1
    
    def load_model(self, model_path, device='cuda:0'):
        # self.model = torch.jit.load(model_path)
        self.model.load_state_dict(torch.load(model_path))
        print('model loaded')
        self.device = device
        self.model = self.model.to(self.device)

    def predict(self, inp):
        self.model.eval()
        with torch.inference_mode():
            inp = inp.to(self.device)
            out = self.model(inp, src_mask=self.src_mask)
        return out

    def runner(self, data_folder, save_folder, epochs=100, train_test_split=0.95, train_batch_size=32, val_batch_size=32, learning_rate=1e-4, device='cuda:0', print_every=50, eval_every=250, save_every=500):

        train_data = []
        for df in data_folder[:-1]:
            with open(df, 'rb') as f:
                train_data += pickle.load(f)
            # balanced_data = glob(data_folder)
        # random.shuffle(balanced_data)
        print('# trajectories:', len(train_data))

        val_data = []
        for df in data_folder[-1:]:
            with open(df, 'rb') as f:
                val_data += pickle.load(f)
            # balanced_data = glob(data_folder)
        # random.shuffle(balanced_data)
        print('# trajectories:', len(val_data))
    
        wandb.init(project='scene_predictor', entity='dm1487')
        save_path = f'{save_folder}/checkpoints'
        self.device = device
        self.model = self.model.to(self.device)
        if self.src_mask is not None:
            self.src_mask = self.src_mask.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # num_train_envs = int(len(balanced_data) * train_test_split)
        # train_idxs = np.arange(0, num_train_envs).astype(int).tolist()
        # val_idxs = np.arange(num_train_envs, len(balanced_data)).astype(int).tolist()
        training_files = train_data # [balanced_data[i] for i in train_idxs]
        val_files = val_data # [balanced_data[i] for i in val_idxs]


        for epoch in range(epochs):
            # if len(training_files) > 128000:
            #     sub_training_files = random.sample(training_files, 128000)
            # else:
            sub_training_files = training_files

            # sub_training_files = training_files
            
            if len(val_files) > 4000:
                sub_val_files = random.sample(val_files, 4000)
            else:
                sub_val_files = val_files

            train_ds = SceneEstimatorDataset(files=sub_training_files, sequence_length=self.sequence_length, estimate_pose=True, start=self.start)
            train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
            
            # sub_val_files = random.sample(val_files, 4000)
            val_ds = SceneEstimatorDataset(files=sub_val_files, sequence_length=self.sequence_length, estimate_pose=True, start=self.start)
            val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=True)

            val_loss = self.train(train_dl, val_dl, save_folder, print_every=print_every, eval_every=eval_every, save_every=save_every)

            PATH = f'{save_path}/model_{self.saved_models}.pt'
            torch.save(self.model.state_dict(), PATH)
            self.saved_models += 1
            print(f'{time.time()} model saved at {PATH}')

    def load_pretrained(self, model_path, fine_tune=True):
        from model import PretrainTransformer
        self.pretrain_model = PretrainTransformer(input_size=39, output_size=39, embed_size=512, hidden_size=2048, num_heads=4, max_sequence_length=self.sequence_length, num_layers=8)
        self.pretrain_model.load_state_dict(torch.load(model_path))
        print('pretrain model loaded')
        if fine_tune:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
            self.pretrain_model.eval()

        class SceneLayer(nn.Module):
            def __init__(self, pretrained_model):
                super(SceneLayer, self).__init__()
                self.pretrained_model = pretrained_model
                self.scene_layers = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 21))
            
            def forward(self, x, src_mask):
                x = self.pretrained_model.linear_in(x)
                x = self.pretrained_model.positonal_embedding(x)
                x = self.pretrained_model.encoder(x, mask=src_mask)
                x = self.pretrained_model.linear_out(x)
                x = self.scene_layers(x)
                return x
            
        self.model = SceneLayer(self.pretrain_model)
        print(self.model)

if __name__ == '__main__':
    
    import os

    # data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/1_obs/8/*/*.npz'
    # model_file = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-08-30_16-39-37/checkpoints/model_1.pt'
    # se = SceneEstimator()
    # se.load_model(model_file)
    # m = torch.jit.script(se.model)
    # torch.jit.save(m, '/common/users/dm1487/test_model_1.pt')

    save_folder = f'/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/final_results_2_obs_se/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    # data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/test/0/all_files.pkl'
    
    data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep2/2_obs/all_files_bal/all_files_1.pkl'
    
    # data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep6/2_obs/all_files_bal/all_files_0.pkl'

    data_folder = '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep7/2_obs/all_files_bal/all_files_balanced_0_1_2.pkl'

    data_folders = [
        '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep16/2_obs/all_files_bal/all_files_1_train.pkl',
        '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep7/2_obs/all_files_bal/all_files_balanced_0_1_2.pkl',
        '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep2/2_obs/all_files_bal/all_files_1.pkl',

        '/common/users/dm1487/legged_manipulation_data_store/trajectories/icra_data_sep16/2_obs/all_files_bal/all_files_1_val.pkl'
    ]
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(f'{save_folder}/checkpoints')
        os.makedirs(f'{save_folder}/viz')

    load_pretrained_model = False
    fine_tune = False
    device = 'cuda:0'
    input_size = 39 # - 12 + 3
    num_layers = 2
    num_heads = 2
    bs = 64
    hidden_size = 1024
    embed_size = 512
    causal = True # torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
    se = SceneEstimator(input_size=input_size, causal=causal, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, start=0)
    if load_pretrained_model:
        model_path = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/pretrain_2_obs/2023-08-30_22-18-10/checkpoints/model_state_dict11.pt'
        se.load_pretrained(model_path, fine_tune=fine_tune)

    with open(f'{save_folder}/info.txt', 'w') as f:
        f.write(f'final training {"causal" if causal else "not causal"}, with pose, {input_size}, {"yes" if load_pretrained_model else "no"} pretrain, split loss, {num_layers} layers, {num_heads} heads, mse loss, alternate_fn2, all data 1 (equal), with confidence bit')

    se.runner(data_folders, save_folder, epochs=50, train_test_split=0.95, train_batch_size=bs, val_batch_size=bs, learning_rate=1e-4, device='cuda:0', print_every=250, eval_every=500, save_every=500)
