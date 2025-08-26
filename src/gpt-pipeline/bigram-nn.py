import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits (scores) for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both tensor of integers
        logits = self.token_embedding_table(idx) # (Batch by time by channel) tensor (B, T, C)

        # idx and targets are both (B,T) tensor of integers 
        if targets is None:
            loss = None
        else:
            # Reshape and cross entropy calculations for loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # modify tensor for C in second dimension (its a pytorch thing for how they calculate cross entropy)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # we have the identity of the next character, how well are we predicting it?
        
        # Expected loss of -ln(1/65) for TinyShakespeare (or 4.174...), from my testing we achieve about 4.875-5.0 which means we guess wrong a bit (8-26-25)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indicies in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # focus on last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)a
        return idx

