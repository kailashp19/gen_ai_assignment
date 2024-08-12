from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
import numpy as np
from sentence_transformers import SentenceTransformer

# Connect to the Qdrant service
client = QdrantClient("http://localhost:6333")

# Define the dimensionality of your embeddings
vector_size = 384  # Replace with the actual size of your embeddings

# Load a pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create a collection to store the embeddings
client.recreate_collection(
    collection_name="embeddings_collection",
    vectors_config=VectorParams(size=vector_size, distance="Cosine")
)

# Example embeddings and corresponding IDs
embeddings = np.load('sentence_embeddings.npy')  # Load your embeddings
ids = list(range(len(embeddings)))  # Create IDs for each embedding

# Insert embeddings into the Qdrant collection
client.upload_collection(
    collection_name="embeddings_collection",
    vectors=embeddings,
    ids=ids
)

# Encode the query into an embedding
query = "Marathon in athletes"
query_embedding = model.encode([query]).flatten()  # Flatten the query embedding
print(f"Query embeddings from {query} is: {query_embedding}")

# Perform the search
search_results = client.search(
    collection_name="embeddings_collection",
    query_vector=query_embedding,
    limit=3  # Number of nearest neighbors to return
)

# Retrieve the most similar sentences based on their IDs
for result in search_results:
    print(f"ID: {result.id}, Score: {result.score}")
