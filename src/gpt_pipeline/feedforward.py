import parameters

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """ Linear Layer followed by a non-linearity layer """

    # n_embd from Attention Mask is: Number of embedding dimensions

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(parameters.dropout),
        )

    # Torch does all the work for us honestly, thisll be quite a large milestone if we can code this ourselves

    def forward(self, x):
        return self.net(x)