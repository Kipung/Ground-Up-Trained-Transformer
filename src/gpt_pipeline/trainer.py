# Order: Characterization -> Tokenization -> Encode

# Tokenize using a basic Schema (character driven cause we're keeping it simple)

# Here is the data set link: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Lame Torch stuff that we want to eventually code by hand
import parameters

import time
import sys

import torch
import torch.nn as nn

class Dataset:
    def __init__(self, file_path="./src/gpt_pipeline/TinyShakespeare.txt", block_size=parameters.block_size, batch_size=parameters.batch_size, debug=False):
        self.debug = debug
        self.block_size = block_size # Context Length for Predictions (maximum)
        self.batch_size = batch_size # how many independent sequences will we process in parallel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = 3e-4
        self.eval_iterations = 200
        self.eval_interval = 200
        self.max_iterations = 2000

        # set the seeding
        torch.manual_seed(3465)

        # load dataset in proper encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        if self.debug:
            print("Length of Dataset (Characters): ", len(self.text))

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        # this will print all the characters in the corpus as well as the amount
        if self.debug:
            print(''.join(self.chars))
            print(self.vocab_size)

        # Encoding / decoding maps
        stoi = {ch: i for i, ch in enumerate(self.chars)}
        itos = {i: ch for i, ch in enumerate(self.chars)}

        # Encoding / decoding functions
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: ''.join([itos[i] for i in l])

        # Encode dataset into torch tensor (We want to make our own tensor class eventually)
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)

        # We want to train 90% of the data and use 10% of the data to validate the trained data
        n = int(0.9*len(self.data)) # 90% of the data
        self.trainer_data = self.data[:n]
        self.validation_data = self.data[n:]

        # this is the chunk of data we will train in an instant (this improves performance drastically (not all data will be trained at once))
        # trainer_data[:block_size+1]

def train(Dataset: Dataset, model: nn.Module, device_model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=Dataset.learning_rate)
    params = sum(p.numel() for p in device_model.parameters())/1e6
    print(params, 'M parameters')

    start_time = time.time()
    avg_time = None
    last_print = 0
    bar_length = 40  # number of characters in the progress bar

    for iter in range(Dataset.max_iterations):
        iter_start = time.time()

        # sample a batch of data
        xb, yb = get_batch(Dataset, 'trainer')

        # evaluate loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # update avg iteration time
        iter_time = time.time() - iter_start
        avg_time = iter_time if avg_time is None else 0.95 * avg_time + 0.05 * iter_time

        # print once per second
        now = time.time()
        if now - last_print >= 1.0:
            iters_done = iter + 1
            iters_left = Dataset.max_iterations - iters_done
            eta_seconds = avg_time * iters_left
            eta_mins = int(eta_seconds // 60)
            eta_secs = int(eta_seconds % 60)

            # progress bar
            progress = iters_done / Dataset.max_iterations
            filled = int(bar_length * progress)
            bar = '=' * filled + '-' * (bar_length - filled)

            # losses only every eval_interval
            loss_str = ""
            if iter % Dataset.eval_interval == 0:
                losses = estimate_loss(Dataset, model)
                loss_str = f" | train {losses['trainer']:.4f}, val {losses['validation']:.4f}"

            # print in-place
            sys.stdout.write(
                f"\r[{bar}] {progress*100:5.1f}% "
                f"| Step {iters_done}/{Dataset.max_iterations} "
                f"| ETA {eta_mins}m {eta_secs}s{loss_str}"
            )
            sys.stdout.flush()

            last_print = now

    print()  # move to new line after finishing
    torch.save(device_model.state_dict(), f"./src/weights/trained_data{params}")
    

def push_data(Dataset: Dataset, device_model):
    context = torch.zeros((1,1), dtype=torch.long, device=Dataset.device)
    print(Dataset.decode(device_model.generate(context, max_new_tokens=500)[0].tolist()))

def estimate_loss(Dataset: Dataset, model: nn.Module):
    out = {}
    model.eval()
    for split in ['trainer', 'validation']:
        losses = torch.zeros(Dataset.eval_iterations)
        for k in range(Dataset.eval_iterations):
            X, Y = get_batch(Dataset, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_batch(Dataset: Dataset, split="trainer"):
    # generate a small batch of data of inputs (x) and possible targets (y)
    data = Dataset.trainer_data if split == "trainer" else Dataset.validation_data
    ix = torch.randint(len(data) - Dataset.block_size, (Dataset.batch_size,))
    x = torch.stack([data[i:i+Dataset.block_size] for i in ix])
    y = torch.stack([data[i+1:i+Dataset.block_size+1] for i in ix])
    x, y = x.to(Dataset.device), y.to(Dataset.device)
    return x, y