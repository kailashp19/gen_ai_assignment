from sentence_transformers import SentenceTransformer
import numpy as np

# Load the text document
with open('text_files/Paris2024-QS-Athletics.txt', 'r', encoding='utf-8') as file:
    sentences = [line.strip() for line in file.readlines() if line.strip()]

# Load a pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(sentences, show_progress_bar=True)

# Convert to a NumPy array
embeddings = np.array(embeddings)

# Save embeddings to a file
np.save('sentence_embeddings.npy', embeddings)

# Save sentences for reference
with open('sentences.txt', 'w', encoding='utf-8') as file:
    for sentence in sentences:
        file.write(f"{sentence}\n")

print("Embeddings generated and saved.")
print(f"Length of the embeddings: {len(embeddings[0])}")