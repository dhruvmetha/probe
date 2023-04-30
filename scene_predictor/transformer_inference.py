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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
window_size = 250
sequence_length = 500
hidden_state_size = 2048
num_heads = 8
num_layers = 4
alg = 'transformer'
eval_every = 250
print_every = 50
epochs = 100
train_batch_size = 64
test_batch_size = 64
learning_rate = 1e-4
dropout = 0.
input_size = 70
output_size = 27
# RECTS = 2


SAVE_FOLDER = Path(f'./scene_predictor/results/{alg}_{sequence_length}_{hidden_state_size}')
# SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
PLOT_FOLDER = 'plots_eval'
CHECKPOINT_FOLDER = 'checkpoints'
CHECKPOINT_FILE = sorted(glob(f'{SAVE_FOLDER}/{CHECKPOINT_FOLDER}/*.pt'), key= lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
traj_data_file = '/common/users/dm1487/legged_manipulation_data/rollout_data/nsf/single_trajectories'
all_train_test_files = sorted(glob(f'{traj_data_file}/*/*.npz'))


num_train_envs = int(len(all_train_test_files) * 0.0)
train_idxs = np.arange(0, num_train_envs).astype(int).tolist()
val_idxs = np.arange(num_train_envs, len(all_train_test_files)).astype(int).tolist()
training_files = [all_train_test_files[i] for i in train_idxs]
val_files = [all_train_test_files[i] for i in val_idxs]

print(CHECKPOINT_FILE)
# train_ds = TransformerDataset(files=training_files, sequence_length=sequence_length)
# train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

val_ds = TransformerDataset(files=val_files, sequence_length=sequence_length)
val_dl = DataLoader(val_ds, batch_size=test_batch_size, shuffle=True)


# print(len(train_ds), len(val_ds))

# inp, target, mask, fsw = next(iter(train_dl))


model = MiniTransformer(input_size=input_size, output_size=output_size, embed_size=128, hidden_size=hidden_state_size, num_heads=8, max_sequence_length=250, num_layers=num_layers)

model.load_state_dict(torch.load(CHECKPOINT_FILE))
model = model.to(device)
model.eval()

def loss_fn(out, targ, mask):

    k = 6

    loss1 = F.binary_cross_entropy(torch.sigmoid(out[:, :, k:k+1]), targ[:, :, k:k+1], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+7:k+8]), targ[:, :, k+7:k+8], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+14:k+15]), targ[:, :, k+14:k+15], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, 21:22]), targ[:, :, 21:22], reduction='none')

    loss2 = F.binary_cross_entropy(torch.sigmoid(out[:, :, k+1:k+2]), targ[:, :, k+1:k+2], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+8:k+9]), targ[:, :, k+8:k+9], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, k+15:k+16]), targ[:, :, k+15:k+16], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, 22:23]), targ[:, :, 22:23], reduction='none')


    loss3 = F.mse_loss(out[:, :, k+2:k+4], targ[:, :, k+2:k+4], reduction='none') + F.mse_loss(out[:, :, k+9:k+11], targ[:, :, k+9:k+11], reduction='none') + F.mse_loss(out[:, :, k+16:k+18], targ[:, :, k+16:k+18], reduction='none') # + F.mse_loss(out[:, :, 23:25], targ[:, :, 23:25], reduction='none')
    loss3 = torch.sum(loss3, dim=-1).unsqueeze(-1)

    loss4 = F.mse_loss(out[:, :, k+4:k+7], targ[:, :, k+4:k+7], reduction='none') + F.mse_loss(out[:, :, k+11:k+14], targ[:, :, k+11:k+14], reduction='none') + F.mse_loss(out[:, :, k+18:k+21], targ[:, :, k+18:k+21], reduction='none') # + F.mse_loss(out[:, :, 25:], targ[:, :, 25:], reduction='none')
    loss4 = torch.sum(loss4, dim=-1).unsqueeze(-1)

    loss_pose = F.mse_loss(out[:, :, :k], targ[:, :, :k], reduction='none')
    loss_pose = torch.sum(loss_pose, dim=-1).unsqueeze(-1)

    
    return loss1, loss2, loss3, loss4, loss_pose

src_mask = torch.triu(torch.ones(250, 250) * float('-inf'), diagonal=1).to(device)
all_anim = []
patches_ctr = 0
val_total_loss = 0
anim_idx = {

}
anim_ctr = 0

print(len(val_ds))
with torch.inference_mode():
    model.eval()
    for i, (inp, targ, mask, fsw) in tqdm(enumerate(val_dl)):
        print(i)
        # if i not in anim_idx:
        #     anim_idx
        inp = inp.to(device)
        targ = targ.to(device)
        mask = mask.to(device)
        fsw = fsw.to(device)

        anim_idx = 0 # min(anim_idx, inp.shape[0]-1)

        out = model(inp, src_mask=src_mask)

        loss1, loss2, loss3, loss4, loss_pose = loss_fn(out, targ, mask)
        loss = (loss1 + loss2 + loss3 + loss4 + loss_pose)
        loss = torch.sum(loss*mask)/torch.sum(mask)

        val_total_loss += loss.item()

        # if anim_ctr < 10:
        patches = []
        # print(inp.shape[1])
        for step in range(inp.shape[1]):
            # print(mask[anim_idx, step, 0])
            if mask[anim_idx, step, 0]:
                patch_set = get_visualization(anim_idx, targ[:, step, :6].squeeze(1), targ[:, step, 6:].squeeze(1), out[:, step, :6].squeeze(1), out[:, step, 6:].squeeze(1), fsw[:, step, :].squeeze(1))
                patches.append(patch_set)
        all_anim.append(patches)
            # anim_ctr += 1

path = SAVE_FOLDER/f'{PLOT_FOLDER}'
path.mkdir(parents=True, exist_ok=True)

# print(len(all_anim), len(all_anim[0]))

for local_idx, anim in enumerate(all_anim):
    # print(len(anim))
    with open(path/f'plot_{patches_ctr+local_idx}.pkl', 'wb') as f:
        pickle.dump(anim, f)
patches_ctr += len(all_anim)
all_anim = []





