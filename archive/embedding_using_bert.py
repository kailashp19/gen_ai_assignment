import torch 
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

# Load text data
file = 'text_files/Paris2024-QS-Athletics.txt'
with open(file, 'r', encoding='utf-8') as f:
    words = f.read()

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize(words)
vocab_size = tokenizer.vocab_size
word_ids = tokenizer.convert_tokens_to_ids(tokens)

# Padding the sequences
sequence_length = 512
sequences = [word_ids[i:i+sequence_length] for i in range(0, len(word_ids), sequence_length)]
# Pad the last sequence if it's shorter than the desired length
if len(sequences[-1]) < sequence_length:
    sequences[-1] = sequences[-1] + [tokenizer.pad_token_id] * (sequence_length - len(sequences[-1]))

class TextDatasets(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        input_tokens = self.sequences[index]
        return torch.tensor(input_tokens, dtype=torch.long)

dataset = TextDatasets(sequences)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # Scaling the embeddings

# Instantiate the embedding model
embedding_model = Embedding(vocab_size)

# Prepare to save embeddings
embedding_list = []

# Process each batch of data and generate embeddings
for batch in dataloader:
    input_tokens = batch
    embeddings = embedding_model(input_tokens)
    embedding_list.append(embeddings.detach().numpy())

# Convert the list to a numpy array
embedding_array = np.concatenate(embedding_list, axis=0)

# Save embeddings to a file (e.g., NumPy binary format)
output_file = 'embeddings_from_bert_transformers.npy'
np.save(output_file, embedding_array)
print(f"Embeddings saved to {output_file}")

# Load pre-trained BERT model for QnA
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Function to answer questions based on the text
def answer_question(question, context, max_length=512, stride=128, confidence_threshold=0.1):
    # Tokenize question to calculate its length
    question_tokens = tokenizer(question, return_tensors='pt').input_ids[0]
    question_length = len(question_tokens)

    # Calculate context chunk size dynamically
    context_chunk_size = max_length - question_length - 3  # 3 for special tokens [CLS], [SEP]

    # Tokenize context into tokens
    tokens = tokenizer(context, return_tensors='pt', add_special_tokens=False).input_ids[0]
    
    # Split the context into overlapping chunks
    chunks = [tokens[i:i + context_chunk_size] for i in range(0, len(tokens), context_chunk_size - stride)]
    
    best_answer = ""
    best_score = -float('inf')
    
    for chunk in chunks:
        # Encode question and chunked context with truncation
        inputs = tokenizer.encode_plus(
            question,
            chunk.tolist(),
            return_tensors='pt',
            truncation='only_second',  # Truncate only the context, not the question
            max_length=max_length,
            padding='max_length',
            return_overflowing_tokens=False
        )
        
        # Get start and end logits
        with torch.no_grad():
            outputs = model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

        # Find the max start and end logits
        start_score, start_index = torch.max(answer_start_scores, dim=1)
        end_score, end_index = torch.max(answer_end_scores, dim=1)
        total_score = (start_score + end_score).item() / 2

        if total_score > confidence_threshold and total_score > best_score:
            best_score = total_score
            answer_start = start_index.item()
            answer_end = end_index.item() + 1

            best_answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end], skip_special_tokens=True)
    
    return best_answer if best_answer.strip() else "No answer found."

# Example question
question = "What do you mean by reserved athletes?"
context = words  # This can be a smaller chunk if needed

# Get the answer
answer = answer_question(question, context)
print(f"Answer: {answer}")