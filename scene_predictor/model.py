from torch import nn
import torch
import math

class MiniTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_obstacles=1, embed_size=512, hidden_size=2048, num_heads=8, max_sequence_length=250, num_layers=6, estimate_pose=True, layer_norm=True):
        super(MiniTransformer, self).__init__()

        self.batch_first = True

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_in = nn.Linear(input_size, embed_size)
        self.positonal_embedding = PositionalEncoding(embed_size, max_len=max_sequence_length)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=self.batch_first)
        
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        
        self.activation = nn.ELU()
        

        # self.linear_out = nn.Linear(embed_size, embed_size)
        self.linear_out = nn.Linear(embed_size, output_size*num_obstacles)

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm_in = nn.LayerNorm(embed_size)
            self.layer_norm_out = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(0.2)

        self.out = nn.ModuleList([nn.Sequential(nn.Linear(embed_size, 64), nn.ELU(), nn.Dropout(0.2), nn.Linear(64, output_size)) for _ in range(num_obstacles)])
        self.estimate_pose = estimate_pose
    
    def forward(self, x, src_mask=None):
        x = self.linear_in(x)
        if self.layer_norm:
            x = self.layer_norm_in(x)
        x = self.dropout(x)
        x = self.positonal_embedding(x)
        if src_mask is None:
            x = self.encoder(x)
        else:
            x = self.encoder(x, mask=src_mask)
        x = self.linear_out(x)
        return x
        if self.layer_norm:
            x = self.layer_norm_out(x)
        x = self.dropout(x)
        x = self.activation(x)
        priv_info = [out(x) for out in self.out]
        return torch.cat(priv_info, dim=-1)
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)
    
class GRUPose(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, num_layers=1, dropout=0.2):
        super(GRUPose, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation = nn.ELU()
    
    def forward(self, x, hidden):
        all_hidden_states, fh = self.gru(x, hidden)
        x = self.activation(all_hidden_states)
        x = self.linear(x)
        return x, fh, all_hidden_states


class PretrainTransformer(nn.Module):
    def __init__(self, input_size, output_size, embed_size=512, hidden_size=2048, num_heads=8, max_sequence_length=250, num_layers=6):
        super(PretrainTransformer, self).__init__()

        self.batch_first = True

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_in = nn.Linear(input_size, embed_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.linear_out = nn.Linear(embed_size, embed_size)

        self.out = nn.Sequential(nn.Linear(embed_size, 128), nn.ELU(), nn.Linear(128, output_size))
        self.positonal_embedding = PositionalEncoding(embed_size, max_len=max_sequence_length)
        self.activation = nn.ELU()
    
    def forward(self, x):
        x = self.linear_in(x)
        x = self.positonal_embedding(x)
        x = self.encoder(x)
        x = self.linear_out(x)
        x = self.activation(x)
        x = self.out(x)
        return x
    

# class YawToRotationMatrix(nn.Module):
#     def forward(self, yaw_batch):
#         # Convert batch of yaw angles to radians
#         yaw_batch = torch.deg2rad(yaw_batch)

#         # Calculate the rotation matrices for the batch
#         R_yaw = torch.stack([torch.stack([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw)],
#                                          [torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)],
#                                          [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)])
#                             for yaw in yaw_batch])

#         return R_yawyaw


class ScenePoseTransformer(nn.Module):
    def __init__(self, input_size, output_size, embed_size=512, hidden_size=2048, num_heads=8, max_sequence_length=250, num_layers=6, estimate_pose=True):
        super(ScenePoseTransformer, self).__init__()

        self.batch_first = True

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_in = nn.Linear(input_size, embed_size)
        self.positonal_embedding = PositionalEncoding(embed_size, max_len=max_sequence_length)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=self.batch_first)
        
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        
        self.activation = nn.ELU()

        self.linear_out = nn.Linear(embed_size, embed_size)

        self.out_pose = nn.Sequential(nn.Linear(embed_size, 64), nn.ELU(), nn.Linear(64, 2))
        self.out_scene = nn.ModuleList([nn.Sequential(nn.Linear(embed_size, 64), nn.ELU(), nn.Linear(64, 8)) for _ in range(3)])
        self.estimate_pose = estimate_pose
    
    def forward(self, x, src_mask=None):
        x = self.linear_in(x)
        x = self.positonal_embedding(x)
        if src_mask is None:
            x = self.encoder(x)
        else:
            x = self.encoder(x, mask=src_mask)
        x = self.linear_out(x)
        x = self.activation(x)

        pose = self.out_pose(x)
        priv_info = [out(x) for out in self.out_scene]
        return pose, torch.cat(priv_info, dim=-1)
        

class VelocityEstimatorTransformer(nn.Module):
    def __init__(self, input_size, output_size, embed_size=512, hidden_size=2048, num_heads=8, max_sequence_length=250, num_layers=6, estimate_pose=True):
        super(VelocityEstimatorTransformer, self).__init__()

        self.batch_first = True

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_in = nn.Linear(input_size, embed_size)
        self.layer_norm_in = nn.LayerNorm(embed_size)
        self.positonal_embedding = PositionalEncoding(embed_size, max_len=max_sequence_length)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=self.batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.2)
        self.out_vel = nn.Linear(embed_size, 2)
        self.estimate_pose = estimate_pose
    
    def forward(self, x, src_mask=None):
        x = self.linear_in(x)
        x = self.layer_norm_in(x)
        x = self.dropout(x)
        x = self.positonal_embedding(x)
        if src_mask is None:
            x = self.encoder(x)
        else:
            x = self.encoder(x, mask=src_mask)
        x = self.activation(x)
        vel = self.out_vel(x)
        return vel
        