import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000,
        batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        # self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if self.batch_first:
            # (batch_size, seq_len, embedding_dim)
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            # (seq_len, batch_size, embedding_dim)
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if self.batch_first:
            x = x + self.pe[0, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        # return self.dropout(x)
        return x


class TiedEmbeddingsTransposed(nn.Module):
    def __init__(self, tied_to) -> None:
        super().__init__()
        self.tied_to = tied_to

    def forward(self, x):
        output = torch.matmul(x, self.tied_to.weight.T)
        return output