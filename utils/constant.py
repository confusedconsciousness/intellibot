# --- Configuration ---
SOURCE_DIRECTORY = "knowledge/source"  # Where your .txt, .md, .pdf, .html files are
CHROMA_DB_DIRECTORY = "knowledge/chroma_db"
COLLECTION_NAME = "test"

# Advanced Configuration
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 500
EMBEDDING_MODEL_NAME = "models/embedding-001"  # Google's embedding model
FORCE_RECREATE_STORE = True  # Set to True to re-process and re-store all documents
