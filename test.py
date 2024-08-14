import torch 
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import math


file = 'yourfile-name.txt'
with open(file, 'r', encoding='utf-8') as f:
    words = f.read()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize(words)
vocab_size = tokenizer.vocab_size
word_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Number of tokens: {len(word_ids)}")
print(len(word_ids), len(tokens))


sequence_length = 512
sequences = [word_ids[i:i+sequence_length] for i in range(0, len(word_ids)-sequence_length, sequence_length)]

class TextDatasets(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequences = self.sequences[index]
        input_tokens = sequences[:-1]
        output_tokens = sequences[1:]
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(output_tokens, dtype=torch.long)

dataset = TextDatasets(sequences)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class Embedding(nn.Module):
  def __init__(self,  vocab_size, d_model=512)
  	super(Embedding, self).__init__()
		embedding = nn.Embedding(vocab_size, d_model)
    
  def forward(self, x, mask=None):
  	return self.embeding(x) * math.sqrt(self.d_model) # Scaling the embeddings