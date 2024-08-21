import numpy as np

load_embedding = np.load('embeddings_from_bert_transformers.npy')

print(load_embedding[0])

tokens = tokenizer.tokenize(query)
vocab_size = tokenizer.vocab_size
word_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Number of tokens: {len(word_ids)}")
print(len(word_ids), len(tokens))

sequence_length = 3
sequences = [word_ids[i:i+sequence_length] for i in range(0, len(word_ids)-sequence_length, sequence_length)]

dataset = TextDatasets(sequences)
print(dataset.sequences)
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
flat_doc_embeddings = [embedding.tolist() for embedding in doc_embeddings]

client.delete_collection(collection_name=collection_name)

# Create a collection to store the embeddings
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=512, distance='Cosine')
)

for embedding in flat_doc_embeddings:
    if len(embedding) != 512:
        print(f"Error: Found an embedding of length {len(embedding)}")

# Add points to the collection
points = [
    PointStruct(id=i, vector=doc_embedding, payload={"document": doc})
    for i, (doc_embedding, doc) in enumerate(zip(flat_doc_embeddings, chunks))
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

