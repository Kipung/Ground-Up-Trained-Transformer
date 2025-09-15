import parameters
import sys
sys.path.append("./src")

import torch
import torch.nn as nn
from torch.nn import functional as F

import gpt_pipeline.TransformerBlock as TransformerBlock

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Each token reads off the logits for the next token from a lookup table (torch integrated (I think we can just yoink it))
        self.token_embedding_table = nn.Embedding(vocab_size, parameters.n_embd)
        self.position_embedding_table = nn.Embedding(parameters.block_size, parameters.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock.Block(parameters.n_embd, n_head=parameters.n_head) for _ in range(parameters.n_layer)])
        self.ln_f = nn.LayerNorm(parameters.n_embd) # Final Layer Normal
        self.lm_head = nn.Linear(parameters.n_embd, vocab_size)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        reserve = torch.empty(int(2 * 1024**3 / 4), dtype=torch.float32, device="cuda:0")

        if self.device == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
            print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**2, 1), "MB")
            print("Reserved: ", round(torch.cuda.memory_reserved(0)/1024**2, 1), "MB")

        self.apply(self._init_weights)

    # this is some chatGPT jargain. Im hoping that our version can avoid weird stuff like this
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    # Lots of matrix multiplaction and embedding stuff here, needs some more research
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -parameters.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx    

