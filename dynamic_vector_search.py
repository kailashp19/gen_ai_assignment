from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import nltk
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings

warnings.filterwarnings('ignore')

nltk.download('punkt_tab')

# Connect to the Qdrant service
client = QdrantClient("http://localhost:6333")

# Load model for generating embeddings
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# defining the name of the collection
collection_name = "Paris_Olympics_Gen_AI_Collection"

# Load text from a file
with open('sentences.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Split the document into chunks (sentences in this case)
sentences = sent_tokenize(text)
chunk_size = 3  # Adjust chunk size as needed
chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

# Generate embeddings for each chunk
embeddings = np.load('sentence_embeddings.npy')
print(embeddings.shape[1])

# Create a collection to store the embeddings
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embeddings.shape[1], distance='Cosine')
)

print(embeddings[0])

# Add points to the collection
points = [
    PointStruct(id=i, vector=embedding, payload={"document": doc})
    for i, (embedding, doc) in enumerate(zip(embeddings, chunks))
]

client.upsert(collection_name=collection_name, points=points)

# Encode the query into an embedding
query = "Define reserved athletes in 15 words from the document"
query_embedding = model.encode([query])[0]  # Flatten the query embedding
print(query_embedding)

# Perform the search
search_results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=1  # Number of nearest neighbors to return
)

# Display the search results
print("\nSearch Results:")
for idx, result in enumerate(search_results, start=1):
    text_chunk = result.payload.get("document", "No text found")
    score = result.score
    print(f"{idx}. Response: {text_chunk}\n   Score: {score:.4f}\n")