from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, CreateCollection
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import warnings
from embedding_using_bert import Embedding, TextDatasets
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')

# Connect to the Qdrant service
client = QdrantClient("http://localhost:6333")

# Load model for generating embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# defining the name of the collection
collection_name = "Paris_Olympics_Gen_AI_Collection_from_Scratch"

# Load model for generating embeddings
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Load text from a file
with open('sentences.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# getting the phrases as a list
text_list = text.split()

# Define chunk size (number of words per chunk)
chunk_size = 3  # Example: 100 words per chunk

# Split words into chunks
chunks = [text_list[i:i + chunk_size] for i in range(0, len(text_list)-chunk_size, chunk_size)]

# Now `chunks` contains the words from the file split into chunks
print(f"Number of chunks: {len(chunks)}")
print(chunks)
print(chunks[:2])  # Print the first two chunks as an example

# Generate embeddings for each chunk
doc_embeddings = np.load('embeddings_from_bert_transformers.npy')
print("Embeddings shape: ", doc_embeddings.shape[0])

# passing the query
query = "Who are reserved athletes"

# tokenize the text from the text file
tokens = tokenizer.encode(query, add_special_tokens=True)
max_length = min(len(tokens), 512)
print(max_length)
tokens = tokens[:max_length]
print("Tokens: ", tokens)

# converting 2D array to 1D
fde = doc_embeddings.flatten()

# Create a collection to store the embeddings
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=fde.shape[0], distance='Cosine')
)

# Add points to the collection
points = [
    PointStruct(id=i, vector=embedding, payload={"document": doc})
    for i, (embedding, doc) in enumerate(zip(fde, chunks))
]