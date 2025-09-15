
# Main Database parameters
block_size = 64 # allow longer sequences
batch_size = 256

# GPT parameters
dropout = 0.2 # less dropout if dataset is large
n_embd = 512 # Number of embedding dimensions (# double hidden size)
n_head = 8 # more attention heads
n_layer = 6 # deeper model