from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, CreateCollection
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import warnings
from embedding_using_bert import Embedding, TextDatasets
warnings.filterwarnings('ignore')

# Connect to the Qdrant service
client = QdrantClient("http://localhost:6333")

# Load model for generating embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# defining the name of the collection
collection_name = "Paris_Olympics_Gen_AI_Collection_from_Scratch"

# Load text from a file
with open('sentences.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# getting the phrases as a list
text_list = text.split()

# Define chunk size (number of words per chunk)
chunk_size = 100  # Example: 100 words per chunk

# Split words into chunks
chunks = [text_list[i:i + chunk_size] for i in range(0, len(text_list)-chunk_size, chunk_size)]

# Now `chunks` contains the words from the file split into chunks
print(f"Number of chunks: {len(chunks)}")
print(chunks[:2])  # Print the first two chunks as an example

# Generate embeddings for each chunk
doc_embeddings = np.load('embeddings_from_bert_transformers.npy')
es = doc_embeddings.shape[1]
print("Embeddings shape: ", doc_embeddings.shape[1])

# passing the query
query = "Who are reserved athletes"
# tokenize the text from the text file
tokens = tokenizer.tokenize(query)
vocab_size = tokenizer.vocab_size
word_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Number of tokens: {len(word_ids)}")
print(len(word_ids), len(tokens))

sequence_length = 3
sequences = [word_ids[i:i+sequence_length] for i in range(0, len(word_ids)-sequence_length, sequence_length)]

dataset = TextDatasets(sequences)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the embedding model
embedding_model = Embedding(vocab_size)

# Prepare to save embeddings
embedding_list = []

# Process each batch of data and generate embeddings
for batch in dataloader:
    input_tokens, output_tokens = batch
    q_embeddings = embedding_model(input_tokens)
    embedding_list.append(q_embeddings.detach().numpy())

# Example: Reducing the embedding size by slicing (if acceptable)
doc_embeddings = [embedding[:511] for embedding in doc_embeddings]

# Create a collection to store the embeddings
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=es, distance='Cosine')
)

print("Dpocument embeddings: ", doc_embeddings[0])

# Add points to the collection
points = [
    PointStruct(id=i, vector=doc_embedding, payload={"document": doc})
    for i, (doc_embedding, doc) in enumerate(zip(doc_embeddings, chunks))
]

client.upsert(collection_name=collection_name, points=points)

# Perform the search
search_results = client.search(
    collection_name=collection_name,
    query_vector=q_embeddings,
    limit=1  # Number of nearest neighbors to return
)

# Display the search results
print("\nSearch Results:")
for idx, result in enumerate(search_results, start=1):
    text_chunk = result.payload.get("document", "No text found")
    score = result.score
    print(f"{idx}. Response: {text_chunk}\n   Score: {score:.4f}\n")

