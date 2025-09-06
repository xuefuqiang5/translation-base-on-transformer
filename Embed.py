import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module): 
    def __init__(self, vocab_size, d_model): 
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, X): 
        return self.embed(X)


class PositonalEncoder(nn.Module): 
    def __init__(self, d_model, max_seq_len=200, dropout=0.1): 
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0 / d_model)))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, X):
        X = X * math.sqrt(self.d_model)
        X = X + self.pe[:, :X.size(1), :].to(X.device)
        return self.dropout(X)