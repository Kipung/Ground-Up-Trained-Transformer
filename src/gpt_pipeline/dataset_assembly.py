import parameters
import torch

class Dataset:
    def __init__(self, file_path="./src/training_data/TinyShakespeare.txt", block_size=parameters.block_size, batch_size=parameters.batch_size, debug=False):
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