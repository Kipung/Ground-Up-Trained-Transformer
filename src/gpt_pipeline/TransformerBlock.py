import parameters
import sys
sys.path.append("./src")

import torch
import torch.nn as nn

import gpt_pipeline.attention_masking as AM
import gpt_pipeline.feedforward as FF

class Block(nn.Module):
    """ Transformer Block: Communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        head_size = n_embd // n_head
        self.sa = AM.MultiHeadAttention(n_head, head_size)
        self.ffwd = FF.FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x