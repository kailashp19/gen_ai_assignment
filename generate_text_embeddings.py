import gensim
from gensim.models import Word2Vec
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load text from a file
with open('text_files/Paris2024-QS-Athletics.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize text into sentences
sentences = sent_tokenize(text)

# Tokenize sentences into words, perform chunking
lines = []
for sentence in sentences:
    words = word_tokenize(sentence)
    # Perform POS tagging (required for chunking)
    tagged_words = nltk.pos_tag(words)
    # Perform chunking using NLTK's ne_chunk
    chunks = ne_chunk(tagged_words)
    print(chunks)
    # Flatten the chunk tree to get a list of words
    flat_chunks = [' '.join(leaf[0] for leaf in chunk.leaves()) if hasattr(chunk, 'label') else chunk[0] for chunk in chunks]
    lines.append(flat_chunks)

# Train Word2Vec model
model = Word2Vec(sentences=lines, vector_size=100, window=5, min_count=1, workers=4)
print(model)

# Generate embeddings for each word in the text
embeddings = []
for line in lines:
    word_embeddings = [model.wv[word] for word in line if word in model.wv]
    if word_embeddings:
        embeddings.append(np.mean(word_embeddings, axis=0))

# Convert list of embeddings to a NumPy array
embeddings = np.array(embeddings)
print("Embeddings: ", embeddings)

# Save embeddings to a file
np.save('word_embeddings.npy', embeddings)

print("Embeddings generated and saved to 'word_embeddings.npy'")