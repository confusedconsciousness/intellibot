import chromadb

# 1. Initialize an in-memory client
client = chromadb.Client()

# 2. Create a collection (if it doesn't exist)
# If you try to create an existing collection, it will raise an error,
collection_name = "my_rag_documents"
collection = client.get_or_create_collection(name=collection_name)

# 3. Define some documents, their metadata, and unique IDs
# In a real RAG system, these would come from your loaded and chunked private data.
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is rapidly advancing.",
    "Machine learning is a subset of AI.",
    "Dogs are loyal companions.",
    "Python is a popular programming language for AI.",
    "In programming, an API gateway is a server that acts as an API front-end. It receives API requests, enforces throttling and security policies, passes requests to the back-end service, and then returns the appropriate response.",
]
metadatas = [
    {"source": "sentence_list", "author": "dev"},
    {"source": "wikipedia", "topic": "AI"},
    {"source": "textbook", "topic": "ML"},
    {"source": "pet_care", "breed": "various"},
    {"source": "programming_guide", "language": "python"},
    {"source": "tech_docs", "topic": "API Gateway"},
]
ids = [f"doc{i}" for i in range(len(documents))]

# For a basic in-memory setup without an external embedding model,
# Chroma will use a default embedding function.
print(f"Adding {len(documents)} documents to the collection...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print("Documents added.")

# 4. Query the collection
query_text = "I am getting error from api"
print(f"\nQuerying for: '{query_text}'")

# Perform a similarity search
results = collection.query(
    query_texts=[query_text],
    n_results=2,  # Get the top 2 most similar results
    # where={"topic": "AI"} # Optional: filter by metadata
)

# Print the results
print("\nRetrieved Results:")
for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    score = results['distances'][0][i]  # Lower distance means higher similarity
    meta = results['metadatas'][0][i]
    print(f"  Document: '{doc}'")
    print(f"  Distance: {score:.4f}")
    print(f"  Metadata: {meta}")
    print("-" * 20)
