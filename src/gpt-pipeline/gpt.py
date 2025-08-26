# Order: Characterization -> Tokenization -> Encode

# Tokenize using a basic Schema (character driven cause we're keeping it simple)

# Here is the data set link: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Lame Torch stuff that we want to eventually code by hand
import torch
import torch.nn as nn
from torch.nn import functional as F

DEBUG_MODE = True

## Parameters
block_size = 64 # Context Length for Predictions (maximum)
batch_size = 256 # how many independent sequences will we process in parallel
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# seeding
torch.manual_seed(3465)

# Open the Corpus in utf-8
with open('./src/gpt-pipeline/TinyShakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

if DEBUG_MODE:
    print("Length of Dataset (Characters): ", len(text))

# this will print all the characters in the corpus as well as the amount
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Encoding characters to integers

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # Take a string, output list of integers associated with character
decode = lambda l: ''.join([itos[i] for i in l]) # Take a list of intergers and output a string

if DEBUG_MODE:
    print(encode("Hello World!"))
    print(decode(encode("Hello World!")))

# Encode dataset into torch tensor (We want to make our own tensor class eventually)
data = torch.tensor(encode(text), dtype=torch.long)

if DEBUG_MODE:
    print(data.shape, data.type)
    print(data[:1000]) # first 1000 characters in encoded view (kinda cool)

## Training

# We want to train 90% of the data and use 10% of the data to validate the trained data
n = int(0.9*len(data)) # 90% of the data
trainer_data = data[:n]
validation_data = data[n:]

# this is the chunk of data we will train in an instant (this improves performance drastically (not all data will be trained at once))
trainer_data[:block_size+1]

def get_batch(split):
    # generate a small batch of data of inputs (x) and possible targets (y)
    data = trainer_data if split == "trainer" else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

