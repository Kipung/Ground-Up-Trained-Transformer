import parameters
import sys
sys.path.append("./src")

import torch
import torch.nn as nn
from torch.nn import functional as F

import gpt_pipeline.TransformerBlock as TransformerBlock

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, device=None):
        super().__init__()

        # Each token reads off the logits for the next token from a lookup table (torch integrated (I think we can just yoink it))
        self.token_embedding_table = nn.Embedding(vocab_size, parameters.n_embd)
        self.position_embedding_table = nn.Embedding(parameters.block_size, parameters.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock.Block(parameters.n_embd, n_head=parameters.n_head) for _ in range(parameters.n_layer)])
        self.ln_f = nn.LayerNorm(parameters.n_embd) # Final Layer Normal
        self.lm_head = nn.Linear(parameters.n_embd, vocab_size)

        self.device = parameters.pick_device(device)

        if self.device == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))

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
    
    def stream(self, idx, max_new_tokens, temperature=1.0):
        """Yield one sampled token id (an int) at a time, given a (1, T) context."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -parameters.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next.item()

    def generate(self, idx, max_new_tokens, temperature=1.0):
        new = list(self.stream(idx, max_new_tokens, temperature))
        return torch.cat((idx, torch.tensor([new], dtype=idx.dtype, device=idx.device)), dim=1)
