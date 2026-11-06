"""Stream text from the trained GUTT checkpoint.

    python generate.py --prompt "ROMEO:" --chars 400 --temperature 0.8 --seed 1
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch

import parameters
from gpt_pipeline.dataset_assembly import Dataset
from gpt_pipeline.gpt_nn import GPTLanguageModel

DEFAULT_WEIGHTS = ROOT / "src" / "weights" / "trained_data19.005505"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--chars", type=int, default=400, help="characters to generate")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = ap.parse_args()

    device = parameters.pick_device(args.device)
    dataset = Dataset(file_path=ROOT / "src" / "training_data" / "TinyShakespeare.txt", device=device)
    unknown = sorted(set(args.prompt) - set(dataset.chars))
    if unknown:
        sys.exit(f"prompt has characters outside the training alphabet: {unknown}")

    model = GPTLanguageModel(dataset.vocab_size, device=device).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    context = torch.tensor([dataset.encode(args.prompt)], dtype=torch.long, device=device)
    sys.stdout.write(args.prompt)
    sys.stdout.flush()
    with torch.no_grad():
        for token in model.stream(context, args.chars, args.temperature):
            sys.stdout.write(dataset.decode([token]))
            sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
