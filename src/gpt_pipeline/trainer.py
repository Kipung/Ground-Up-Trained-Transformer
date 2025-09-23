# Order: Characterization -> Tokenization -> Encode

# Tokenize using a basic Schema (character driven cause we're keeping it simple)

# Here is the data set link: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Lame Torch stuff that we want to eventually code by hand
from gpt_pipeline import dataset_assembly

import time
import sys

import torch
import torch.nn as nn

def train(Dataset: dataset_assembly.Dataset, model: nn.Module, device_model):
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
    

def push_data(Dataset: dataset_assembly.Dataset, device_model):
    context = torch.zeros((1,1), dtype=torch.long, device=Dataset.device)
    print(Dataset.decode(device_model.generate(context, max_new_tokens=500)[0].tolist()))

def estimate_loss(Dataset: dataset_assembly.Dataset, model: nn.Module):
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

def get_batch(Dataset: dataset_assembly.Dataset, split="trainer"):
    # generate a small batch of data of inputs (x) and possible targets (y)
    data = Dataset.trainer_data if split == "trainer" else Dataset.validation_data
    ix = torch.randint(len(data) - Dataset.block_size, (Dataset.batch_size,))
    x = torch.stack([data[i:i+Dataset.block_size] for i in ix])
    y = torch.stack([data[i+1:i+Dataset.block_size+1] for i in ix])
    x, y = x.to(Dataset.device), y.to(Dataset.device)
    return x, y