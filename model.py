import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import Module,ModuleList
from torch.nn import Dropout, LayerNorm
from torch.nn import MultiheadAttention
import copy
import math


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.shared_layer = nn.Linear(args.fea_dim, args.hidden_size)  
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(args.dropout)

        # Define independent predictive heads for each personality dimension
        self.heads = nn.ModuleList([nn.Linear(args.hidden_size, 1) for _ in range(5)])

    def forward(self, x):

        x = x.squeeze()  # (batch_size,1, feature_dim) -> (batch_size, feature_dim)

        x = self.drop(self.relu(self.shared_layer(x)))  
        outputs = [head(x) for head in self.heads]  
        outputs = torch.cat(outputs, dim=1)  #  (batch_size, 5)
        return outputs




