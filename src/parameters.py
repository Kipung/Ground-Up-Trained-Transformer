
# Main Database parameters
block_size = 64 # allow longer sequences
batch_size = 256

# GPT parameters
dropout = 0.2 # less dropout if dataset is large
n_embd = 512 # Number of embedding dimensions (# double hidden size)
n_head = 8 # more attention heads
n_layer = 6 # deeper model


def pick_device(preferred=None):
    import torch
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
