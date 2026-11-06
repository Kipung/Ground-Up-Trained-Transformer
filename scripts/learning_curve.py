"""Train a tiny GUTT from scratch and plot loss plus samples to docs/learning.png.

    python scripts/learning_curve.py            # trains (~3 min CPU), writes docs/learning_run.json
    python scripts/learning_curve.py --replot   # redraw from the saved run
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import parameters

parameters.n_embd, parameters.n_layer, parameters.n_head = 128, 3, 4
parameters.batch_size, parameters.dropout = 32, 0.2

from gpt_pipeline import trainer
from gpt_pipeline.dataset_assembly import Dataset
from gpt_pipeline.gpt_nn import GPTLanguageModel

STEPS, LOG_EVERY, SAMPLE_AT, LR, SEED = 1200, 50, (0, 300, 600, 1200), 1e-3, 1337
PROMPT, SAMPLE_CHARS = "ROMEO:", 120
BG, FG, DIM, ACCENT = "#0d1117", "#e6edf3", "#7d8590", "#58a6ff"


def sample(model, ds):
    model.eval()
    torch.manual_seed(SEED)
    ctx = torch.tensor([ds.encode(PROMPT)], dtype=torch.long, device=ds.device)
    with torch.no_grad():
        text = ds.decode(model.generate(ctx, SAMPLE_CHARS)[0].tolist())
    model.train()
    return text


def train():
    ds = Dataset(batch_size=parameters.batch_size, device="cpu")
    torch.manual_seed(SEED)
    model = GPTLanguageModel(ds.vocab_size, device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    losses, samples, t0 = [], {}, time.time()
    for step in range(STEPS + 1):
        if step in SAMPLE_AT:
            samples[step] = sample(model, ds)
        if step == STEPS:
            break
        xb, yb = trainer.get_batch(ds)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0:
            losses.append((step, loss.item()))
            print(f"step {step:5d} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    losses.append((STEPS, loss.item()))
    return losses, samples, n_params


def plot(losses, samples, n_params, out):
    plt.rcParams.update({"font.family": "sans-serif", "text.color": FG,
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(12, 7.2), facecolor=BG)
    ax = fig.add_axes([0.07, 0.58, 0.9, 0.30], facecolor=BG)
    xs, ys = zip(*losses)
    ax.plot(xs, ys, color=ACCENT, lw=2)
    ax.set_xlabel("training step", color=DIM)
    ax.set_ylabel("train loss (batch)", color=DIM)
    ax.tick_params(colors=DIM)
    ax.grid(color="#21262d")
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    for s in SAMPLE_AT:
        y = dict(losses)[s]
        ax.scatter([s], [y], color="#f0883e", zorder=3)
        ax.annotate(f"step {s}", (s, y), textcoords="offset points", xytext=(10, -4) if s == 0 else (8, 8), color=FG, fontsize=10)
    fig.text(0.07, 0.94, "Watching a tiny GUTT learn Shakespeare", fontsize=17, weight="bold")
    fig.text(0.07, 0.915, f"{n_params / 1e6:.2f}M params (3 layers, 4 heads, n_embd 128), CPU, from scratch, prompt {PROMPT!r}",
             fontsize=10.5, color=DIM)
    for i, s in enumerate(SAMPLE_AT):
        col, row = i % 2, i // 2
        x, y = 0.07 + col * 0.46, 0.47 - row * 0.23
        fig.text(x, y, f"step {s}  (loss {dict(losses)[s]:.2f})", fontsize=11, weight="bold", color="#f0883e")
        fig.text(x, y - 0.015, samples[s][:SAMPLE_CHARS], fontsize=9.5, parse_math=False,
                 family="monospace", va="top", color=FG, wrap=True, linespacing=1.35)
    fig.savefig(out, dpi=140, facecolor=BG)


if __name__ == "__main__":
    out = ROOT / "docs" / "learning.png"
    run = ROOT / "docs" / "learning_run.json"
    out.parent.mkdir(exist_ok=True)
    if "--replot" in sys.argv:
        d = json.loads(run.read_text())
        losses, samples, n_params = [tuple(l) for l in d["losses"]], {int(k): v for k, v in d["samples"].items()}, d["params"]
    else:
        losses, samples, n_params = train()
        run.write_text(json.dumps({"losses": losses, "samples": samples, "params": n_params}, indent=1))
    plot(losses, samples, n_params, out)
    print("wrote", out)
