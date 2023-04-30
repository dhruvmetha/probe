from torch import nn
import torch
import math

# A 2 layer LSTM with 128 hidden units with prediction of every output
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden_states):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, hidden = self.lstm(x, hidden_states)
        out = self.fc(out)
        return out, hidden

# A 2 layer GRU with 128 hidden units with prediction of every output
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden_states):
        out, hidden = self.gru(x, hidden_states)
        out = self.fc(out)
        return out, hidden

class MiniTransformer(nn.Module):
    def __init__(self, input_size, output_size, embed_size=512, hidden_size=2048, num_heads=8, max_sequence_length=250, num_layers=6):
        super(MiniTransformer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_in = nn.Linear(input_size, embed_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.linear_out = nn.Linear(embed_size, output_size)
        self.positonal_embedding = PositionalEncoding(embed_size, max_len=max_sequence_length)
    
    def forward(self, x, src_mask):
        x = self.linear_in(x)
        x = self.positonal_embedding(x)
        x = self.encoder(x, mask=src_mask)
        x = self.linear_out(x)
        return x

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