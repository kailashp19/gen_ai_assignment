import torch 
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import math


file = 'text_files\Paris2024-QS-Athletics.txt'
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
print(sequences)

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
    def __init__(self,  vocab_size, d_model=512):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x, mask=None):
      return self.embedding(x) * math.sqrt(self.d_model) # Scaling the embeddings

# Instantiate the embedding model
embedding_model = Embedding(vocab_size)

# Prepare to save embeddings
embedding_list = []

# Process each batch of data and generate embeddings
for batch in dataloader:
    input_tokens, output_tokens = batch
    embeddings = embedding_model(input_tokens)
    embedding_list.append(embeddings.detach().numpy())

# Convert the list to a numpy array
import numpy as np
embedding_array = np.concatenate(embedding_list, axis=0)

# Save embeddings to a file (e.g., NumPy binary format)
output_file = 'embeddings_from_bert_transformers.npy'
np.save(output_file, embedding_array)
print(f"Embeddings saved to {output_file}")