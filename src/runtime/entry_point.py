import sys
sys.path.append("./src")

import torch

# Methods
from gpt_pipeline import output

# Trainer
from gpt_pipeline import trainer

# Models
from gpt_pipeline import bigram_nn as Bigram
from gpt_pipeline import gpt_nn as GPT

dataset = trainer.Dataset()

gpt_model = GPT.GPTLanguageModel(dataset.vocab_size)
m = gpt_model.to(dataset.device)

trainer.train(dataset, gpt_model, m)

# gpt_model.load_state_dict(torch.load("./src/weights/trained_data2000"))
# gpt_model.eval() # put in reference mode?

# output.line("ROMEO:", dataset, m)