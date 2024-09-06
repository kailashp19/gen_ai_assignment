import faiss
import numpy as np

# Load the embeddings
embeddings = np.load('sentence_embeddings.npy')

# Load the text document (each line should correspond to an embedding)
with open('text_files/Paris2024-QS-Athletics.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Ensure that the number of lines matches the number of embeddings
assert len(lines) == embeddings.shape[0], "Mismatch between number of lines and embeddings."

# Initialize the FAISS index
dimension = embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Add embeddings to the index
index.add(embeddings)

# Example query vector (replace this with an actual embedding)
query = np.random.rand(1, dimension)  # Generate or load your query embedding

# Perform the search
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(query, k)

# Output the results: retrieve corresponding lines from the text file
for i in range(len(indices[0])):
    print(f"Text: {lines[indices[0][i]].strip()}, Distance: {distances[0][i]}")
