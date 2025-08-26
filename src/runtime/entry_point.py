import sys
sys.path.append("./src")

# Trainer
from gpt_pipeline import trainer

# Models
from gpt_pipeline import bigram_nn as Bigram
from gpt_pipeline import gpt_nn as GPT

dataset = trainer.Dataset()
bigram_model = Bigram.BigramLanguageModel(dataset.vocab_size)
m = bigram_model.to(dataset.device)

trainer.push(dataset, bigram_model, m)
