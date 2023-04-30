import torch
from model import MiniTransformer

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

CHECKPOINT_FILE = "/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results/transformer_500_2048/2023-04-30_03-48-00/checkpoints/model_0.pt"
model = MiniTransformer(input_size=input_size, output_size=output_size, embed_size=256, hidden_size=hidden_state_size, num_heads=num_heads, max_sequence_length=sequence_length, num_layers=num_layers)

model.load_state_dict(torch.load(CHECKPOINT_FILE))
model = model.to(device)
model.eval()



src_mask = torch.triu(torch.ones(sequence_length, sequence_length) * float('-inf'), diagonal=1).to(device)
