import sys
import time
sys.path.append("./src")

import torch

# Methods
from gpt_pipeline import output as CLI
from gpt_pipeline.dataset_assembly import Dataset as DA

# Trainer
from gpt_pipeline import trainer

# Models
from gpt_pipeline import gpt_nn as GPT

print(torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = DA(debug=False)

gpt_model = GPT.GPTLanguageModel(dataset.vocab_size)
m = gpt_model.to(device)

# trainer.train(dataset, gpt_model, m)

gpt_model.load_state_dict(torch.load("./src/weights/trained_data19.005505"))
gpt_model.eval() # put in reference mode?

while (True):
    CLI.line("ROMEO:", dataset, m)
    time.sleep(5)