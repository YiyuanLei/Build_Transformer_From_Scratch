import torch
import torch.nn as nn
import math

from torch.utils.backcompat import keepdim_warning


class InputEmbeddings(nn.Module):
    # mapping between a number to a vector of d_model dimensions
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, drop_out: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len # max length of the sentences
        self.drop_out = nn.Dropout(drop_out)

        # create a matrix of dimension (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a position vector of dimension (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin and cos for position terms
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)

        # add a dimension for batch sizes, dimension (1, seq_len, d_model), 1 is batch size
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    def forward(self, x):
        x += (self.pe[:, :x.shape[1], :).requires_grad_(False) # makes pe not learned to save computation costs, x.shape[1] is the sequential length
        # slicing operation[dim1, dim2, dim3], [:,:x,:] means take all elements in dim1, take x length in dime2, and all in deim3
        return self.drop_out(x)
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps # numerical stability to avoid zero standard deviation
        self.alpha = nn.Parameter(torch.ones(1)) # learnable multiplicative parameters
        self.bias = nn.Parameter(torch.zeors(1)) # learnable additive parameters
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim =  True) # keep dimensions easier for broadcasting
        std = x.std(dim = -1, keepdim = True) # dim = -1 , is x.ndim - 1 (last dimension)
        return self.alpha + (x - mean)/(std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))





