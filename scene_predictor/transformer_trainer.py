import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader 
import numpy as np
from glob import glob
import torch
from data import TransformerDataset
from model import MiniTransformer
from tqdm import tqdm

import wandb
from pathlib import Path
from datetime import datetime
from visualization import get_visualization
import pickle
from config import *
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
window_size = 250
sequence_length = 750
hidden_state_size = 2048
embed_size = 512
num_heads = 4
num_layers = 6
alg = 'transformer'
eval_every = 250
test_every = eval_every
save_every = 750
print_every = 50
epochs = 100
train_batch_size = 32
val_batch_size = 32
test_batch_size = 2
learning_rate = 1e-4
dropout = 0.
train_test_split = 0.95
estimate_pose = True
input_size = 27 if estimate_pose else 12 + 3 + 12 + 12 + 6 
output_size = 24 + 6 + 21 # 27
# RECTS = 2

# wandb.init(project='scene_predictor_v1', name=f'{alg}_{sequence_length}_{hidden_state_size}')

SAVE_FOLDER = Path(f'./scene_predictor/results_{"pose" if estimate_pose else "priv_info"}/{alg}_{sequence_length}_{hidden_state_size}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
PLOT_FOLDER = 'plots'
CHECKPOINT_FOLDER = 'checkpoints'
path = SAVE_FOLDER/f'{CHECKPOINT_FOLDER}'
path.mkdir(parents=True, exist_ok=True)
saved_model_idx = 0

################
# with open(f'./scene_predictor/balanced_data_7{"" if estimate_pose else "_priv_info"}.pkl', 'rb') as f:
#     balanced_data = pickle.load(f)

# if not estimate_pose:
#     balanced_data = []
    
# balanced_data += glob('/common/users/dm1487/legged_manipulation/rollout_data_1/final_random_seed_data_1_single_trajectories/*/*.npz')
######################
    # balanced_data += glob('/common/users/dm1487/legged_manipulation/rollout_data_1/random_pos_seed_test_1_single_trajectories/*/*.npz')
# else:

# balanced_data = glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/2_obs/2/*/*.npz')
# balanced_data = glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/2_obs/4/*/*.npz')

# clean_data_folder = Path('/common/users/dm1487/legged_manipulation_data_store/clean_traj_data_1')
# balanced_data_file = clean_data_folder / 'balanced_data_1.pkl'
# with open(balanced_data_file, 'rb') as f:
#     balanced_data = pickle.load(f)

# balanced_data = sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data/exploration_3_single_trajectories1/*/*.npz')) + sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data/exploration_4_single_trajectories1/*/*.npz')) + sorted(glob('/common/users/dm1487/legged_manipulation/rollout_data/exploration_6_single_trajectories1/*/*.npz'))

# traj_data_file = Path(f'/common/users/dm1487/legged_manipulation/rollout_data_1/random_pos_seed_test_1_single_trajectories')
# all_train_test_files = sorted(glob(f'{traj_data_file}/*/*.npz'))

all_train_test_files = glob('/common/users/dm1487/legged_manipulation_data_store/trajectories/1_obs/0/*/*.npz') # balanced_data
random.shuffle(all_train_test_files)

num_train_envs = int(len(all_train_test_files) * train_test_split)
train_idxs = np.arange(0, num_train_envs).astype(int).tolist()
val_idxs = np.arange(num_train_envs, len(all_train_test_files)).astype(int).tolist()
training_files = [all_train_test_files[i] for i in train_idxs]
val_files = [all_train_test_files[i] for i in val_idxs]

# training_files = balanced_data
# test_files = val_balanced_data

train_ds = TransformerDataset(files=training_files, sequence_length=sequence_length, estimate_pose=estimate_pose)
train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

val_ds = TransformerDataset(files=val_files, sequence_length=sequence_length, estimate_pose=estimate_pose)
val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=True)

test_ds = TransformerDataset(files=val_files, sequence_length=sequence_length, estimate_pose=estimate_pose)
test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=True)

print(len(train_ds), len(val_ds), len(test_ds))

# inp, target, mask, fsw = next(iter(train_dl))


model = MiniTransformer(input_size=input_size, output_size=output_size, embed_size=embed_size, hidden_size=hidden_state_size, num_heads=num_heads, max_sequence_length=sequence_length, num_layers=num_layers, estimate_pose=estimate_pose)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dl), epochs=epochs)

def loss_fn(out, targ, mask):

    k = 0 + 0

    if not estimate_pose:
        loss1 = F.binary_cross_entropy(torch.sigmoid(out[:, :, k:k+1]), targ[:, :, k:k+1], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+7:k+8]), targ[:, :, k+7:k+8], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+14:k+15]), targ[:, :, k+14:k+15], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, 21:22]), targ[:, :, 21:22], reduction='none')

        loss2 = F.binary_cross_entropy(torch.sigmoid(out[:, :, k+1:k+2]), targ[:, :, k+1:k+2], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+8:k+9]), targ[:, :, k+8:k+9], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+15:k+16]), targ[:, :, k+15:k+16], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, 22:23]), targ[:, :, 22:23], reduction='none')

        loss3 = F.mse_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4], reduction='none') + F.mse_loss(out[:, :, k+9:k+11], targ[:, :, k+9:k+11], reduction='none') + F.mse_loss(out[:, :, k+16:k+18], targ[:, :, k+16:k+18], reduction='none') # + F.mse_loss(out[:, :, 23:25], targ[:, :, 23:25], reduction='none')
        loss3 = torch.sum(loss3, dim=-1).unsqueeze(-1)

        loss4 = F.mse_loss(out[:, :, k+4:k+7], targ[:, :, k+4:k+7], reduction='none') + F.mse_loss(out[:, :, k+11:k+14], targ[:, :, k+11:k+14], reduction='none') + F.mse_loss(out[:, :, k+18:k+21], targ[:, :, k+18:k+21], reduction='none') # + F.mse_loss(out[:, :, 25:], targ[:, :, 25:], reduction='none')
        loss4 = torch.sum(loss4, dim=-1).unsqueeze(-1)

        loss_pose = 0
    else:
        loss1 = 0
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss_pose = F.mse_loss(out[:, :, :6], targ[:, :, :6], reduction='none')
        loss_pose = torch.sum(loss_pose, dim=-1).unsqueeze(-1)
    
    return loss1, loss2, loss3, loss4, loss_pose

src_mask = torch.triu(torch.ones(sequence_length-1, sequence_length-1), diagonal=1).bool().to(device)
all_anim = []
patches_ctr = 0
for epoch in range(epochs):
    train_total_loss = 0
    current_train_loss = 0

    if len(training_files) > 200000:
        sub_training_files = random.sample(training_files, 200000)
    else:
        sub_training_files = training_files
    
    if len(val_files) > 5000:
        sub_val_files = random.sample(val_files, 5000)
    else:
        sub_val_files = val_files

    train_ds = TransformerDataset(files=sub_training_files, sequence_length=sequence_length, estimate_pose=estimate_pose)
    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

    val_ds = TransformerDataset(files=sub_val_files, sequence_length=sequence_length, estimate_pose=estimate_pose)
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=True)

    print(len(train_ds), len(val_ds))

    model.train()
    for i, (inp, targ, mask, fsw, _) in tqdm(enumerate(train_dl)):
        inp = inp.to(device)
        targ = targ.to(device)
        mask = mask.to(device)

        # print('tr', inp.shape, targ.shape, mask.shape, fsw.shape)

        # new_mask = torch.ones_like(mask) * float('-inf')
        # new_mask[mask.nonzero(as_tuple=True)] = 0.
        # new_mask = new_mask.to(device)

        out = model(inp, src_mask=src_mask)

        loss1, loss2, loss3, loss4, loss_pose = loss_fn(out, targ, mask)
        loss = (loss1 + loss2 + loss3 + loss4 + loss_pose)
        loss = torch.sum(loss*mask)/torch.sum(mask)


        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        optimizer.step()

        current_train_loss += loss.item()
        train_total_loss += loss.item()

        # wandb.log({
        #     'train/contact': (torch.sum(loss1*mask)/torch.sum(mask)).item(),
        #     'train/movable': (torch.sum(loss2*mask)/torch.sum(mask)).item(),
        #     'train/location': (torch.sum(loss3*mask)/torch.sum(mask)).item(),
        #     'train/reconstruction': (torch.sum(loss4*mask)/torch.sum(mask)).item(),
        #     'train/pose': (torch.sum(loss_pose*mask)/torch.sum(mask)).item(),
        # })
        if (i+1)%print_every == 0:
            print(f'step {i+1}:', current_train_loss/print_every)
            current_train_loss = 0

        # with torch.no_grad():

        k = 0
        if (i+1) % eval_every == 0:
            anim_ctr = 0
            val_total_loss = 0
            anim_idx = np.random.randint(0, test_batch_size)
            all_anim = []
            with torch.inference_mode():
                model.eval()
                for idx, (inp, targ, mask, fsw, pose) in tqdm(enumerate(val_dl)):
                    # print('val', inp.shape, targ.shape, mask.shape, fsw.shape)
                    # print(inp.shape[-1])
                    inp = inp.to(device)
                    pred_out = inp.clone()
                    targ = targ.to(device)
                    mask = mask.to(device)
                    fsw = fsw.to(device)
                    pose = pose.to(device)
                    out = model(pred_out, src_mask=src_mask)

                    loss1, loss2, loss3, loss4, loss_pose = loss_fn(out, targ, mask)
                    loss = (loss1 + loss2 + loss3 + loss4 + loss_pose )
                    loss = torch.sum(loss*mask)/torch.sum(mask)

                    val_total_loss += loss.item()

                    # wandb.log({
                    #         'eval/contact': (torch.sum(loss1*mask)/torch.sum(mask)).item(),
                    #         'eval/movable': (torch.sum(loss2*mask)/torch.sum(mask)).item(),
                    #         'eval/location': (torch.sum(loss3*mask)/torch.sum(mask)).item(),
                    #         'eval/reconstruction': (torch.sum(loss4*mask)/torch.sum(mask)).item(),
                    #         'eval/pose': (torch.sum(loss_pose*mask)/torch.sum(mask)).item(),
                    # })

                    if anim_ctr < 5:
                        patches = []
                        # targ = targ[:, :, :].clone()
                        # out[:, :, 6:] *= torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device).unsqueeze(0)

                        out_complete = out.clone() # torch.zeros_like(out)
                        targ_complete = targ.clone() # torch.zeros_like(targ)
                        for step in range(inp.shape[1]):
                            # print(mask[anim_idx, step, 0])
                            # if step == 0:
                            #     targ_complete[:, step, :] += targ[:, step, :]
                            #     out_complete[:, step, :] += out[:, step, :]
                            # else:
                            #     targ_complete[:, step, :] += targ[:, step, :] + targ_complete[:, step-1, :]
                            #     out_complete[:, step, :] += out[:, step, :] + out_complete[:, step-1, :]
                            
                            # targ_complete[:, :, :6] *= torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device).unsqueeze(0)
                            # out_complete[:, :, :6] *= torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device).unsqueeze(0)

                            if mask[anim_idx, step, 0]:
                                # patch_set = get_visualization(anim_idx, targ_complete[:, step, k:k+6]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), targ_complete[:, step, k+6:]* torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device), out_complete[:, step, k:k+6]* torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), out_complete[:, step, k+6:]* torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device), fsw[:, step, :].squeeze(1))

                                if estimate_pose:
                                    ## only pose
                                    patch_set = get_visualization(anim_idx, targ_complete[:, step, :6]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), targ_complete[:, step, k+6:], out_complete[:, step, k:k+6]* torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), out_complete[:, step, k+6:], fsw[:, step, :].squeeze(1), estimate_pose=estimate_pose)
                                else:
                                    ## only priv_info
                                    patch_set = get_visualization(anim_idx, pose[:, step, :]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), targ_complete[:, step, :]*torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device), pose[:, step, :]* torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), out_complete[:, step, :]*torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device), fsw[:, step, :].squeeze(1), estimate_pose=estimate_pose)
                                patches.append(patch_set)
                        all_anim.append(patches)
                        anim_ctr += 1

            scheduler.step(val_total_loss)


            print(f'Epoch: {epoch}, Loss: {train_total_loss/eval_every}, Val Loss: {val_total_loss/len(val_dl)}')
            train_total_loss = 0

            path = SAVE_FOLDER/f'{PLOT_FOLDER}_eval'
            path.mkdir(parents=True, exist_ok=True)

            # print(len(all_anim), len(all_anim[0]))

            for local_idx, anim in enumerate(all_anim):
                # print(len(anim))
                with open(path/f'plot_{patches_ctr+local_idx}.pkl', 'wb') as f:
                    pickle.dump(anim, f)
            patches_ctr += len(all_anim)
            all_anim = []

        if ((i+1) % test_every == 0) and False:
            
            anim_ctr = 0
            all_anim = []
            val_total_loss = 0
            anim_idx = np.random.randint(0, test_batch_size)
            with torch.inference_mode():
                model.eval()
                for idx, (inp, targ, mask, fsw) in tqdm(enumerate(test_dl)):
                    # print('val', inp.shape, targ.shape, mask.shape, fsw.shape)
                    # print(inp.shape[-1])
                    inp = inp.to(device)
                    targ = targ.to(device)
                    mask = mask.to(device)
                    fsw = fsw.to(device)
                    pred_out = torch.zeros_like(inp) # .clone()
                    pred_out[:, 0, :] = inp[:, 0, :]
                    for step in range(1, inp.shape[1]):
                        out = model(pred_out, src_mask=src_mask)
                        pred_out[:, step, :] = inp[:, step, :]
                        pred_out[:, step, -9:-3] = out[:, step-1, k:k+6]
                   
                    anim_idx = min(anim_idx, inp.shape[0]-1)
                    out = model(pred_out, src_mask=src_mask)

                    # loss1, loss2, loss3, loss4, loss_pose = loss_fn(out, targ, mask)
                    # loss = (loss1 + loss2 + loss3 + loss4 + loss_pose)
                    # loss = torch.sum(loss*mask)/torch.sum(mask)

                    # val_total_loss += loss.item()

                    # wandb.log({
                    #         'eval/contact': (torch.sum(loss1*mask)/torch.sum(mask)).item(),
                    #         'eval/movable': (torch.sum(loss2*mask)/torch.sum(mask)).item(),
                    #         'eval/location': (torch.sum(loss3*mask)/torch.sum(mask)).item(),
                    #         'eval/reconstruction': (torch.sum(loss4*mask)/torch.sum(mask)).item(),
                    #         'eval/pose': (torch.sum(loss_pose*mask)/torch.sum(mask)).item(),
                    # })
                    
                    if anim_ctr < 10:
                        patches = []
                        out_complete = out.clone() # torch.zeros_like(out)
                        targ_complete = targ.clone() # torch.zeros_like(targ)
                        for step in range(inp.shape[1]):
                            if mask[anim_idx, step, 0]:
                                patch_set = get_visualization(anim_idx, targ_complete[:, step, k:k+6]*torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), targ_complete[:, step, k+6:]* torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device), out_complete[:, step, k:k+6]* torch.tensor([1/0.33, 1, 3.14, 0.65, 0.65, 0.65], device=device), out_complete[:, step, k+6:]* torch.tensor([1, 1, 1/0.33, 1, 3.14, 1, 1.7] * 3, device=device), fsw[:, step, :].squeeze(1))
                                patches.append(patch_set)
                        all_anim.append(patches)
                        anim_ctr += 1
                    if idx == 5:
                        break
            
            path = SAVE_FOLDER/f'{PLOT_FOLDER}_test'
            path.mkdir(parents=True, exist_ok=True)

            # print(len(all_anim), len(all_anim[0]))

            for local_idx, anim in enumerate(all_anim):
                # print(len(anim))
                with open(path/f'plot_{patches_ctr+local_idx}.pkl', 'wb') as f:
                    pickle.dump(anim, f)
            patches_ctr += len(all_anim)
            all_anim = []

        if (i) % save_every == 0:
            path = SAVE_FOLDER/f'{CHECKPOINT_FOLDER}'
            
            torch.jit.save(torch.jit.script(model), path/f'model_{saved_model_idx}.pt')
            
            # torch.save(model.state_dict(), path/f'model_{saved_model_idx}.pt')
            saved_model_idx += 1

        # wandb.log({
        #     'train/loss': train_total_loss/eval_every,
        #     'eval/loss': val_total_loss/len(val_dl)
        # })
    train_total_loss = 0
    current_train_loss = 0
    val_total_loss = 0
        
torch.save(model.state_dict(), path/f'model_{saved_model_idx}.pt')