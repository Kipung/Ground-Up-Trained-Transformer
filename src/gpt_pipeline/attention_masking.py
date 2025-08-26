import parameters

import torch
import torch.nn as nn
from torch.nn import functional as F

# Parameters
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

class Head(nn.Module):
    """ One Head of Self Attention Masking """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(parameters.block_size, parameters.block_size))) # There needs to be a more efficient way to get the block size from the dataset class

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # input <- (Batch, time-step, channels) or (B, T, C)
        # output -> (Batch, time-step, head size)
        B, T, C = x.shape()
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        # compute the attention scores (this part is like super hard and i barely understand it) (matrix multiplication stuff) (kudos to youtube and chat for this)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # tril gives the lower triangular portion of the matrix
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple Heads of self-attention (in parallel) """

    def __init__(self, num_heads, heads_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(heads_size) for _ in range(num_heads)]) # Fill a list of heads
        self.proj = nn.Linear(heads_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
