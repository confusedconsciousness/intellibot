from rag_pipeline import RAGPipeline
from utils.constant import SOURCE_DIRECTORY, CHROMA_DB_DIRECTORY, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, \
    EMBEDDING_MODEL_NAME, FORCE_RECREATE_STORE

if __name__ == "__main__":
    print("Initializing RAG Pipeline...")
    try:
        pipeline = RAGPipeline(
            source_dir=SOURCE_DIRECTORY,
            chroma_dir=CHROMA_DB_DIRECTORY,
            collection_name=COLLECTION_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            embedding_model_name=EMBEDDING_MODEL_NAME
        )

        print(f"\nSetting up Vector Store (force_recreate={FORCE_RECREATE_STORE})...")
        pipeline.setup_vector_store(force_recreate=FORCE_RECREATE_STORE)

        # Example Query (only if setup was successful or store already existed)
        if pipeline.vector_store_manager.get_collection_count() > 0:
            print("\n💡 Example Usage: Querying the Vector Store")
            sample_query = "What is inverted index"  # Change to a relevant query
            pipeline.query(sample_query, k=2)

            print("\n💡 Example Usage: Querying with a different query")
            sample_query = "What is API Gateway?"
            pipeline.query(sample_query, k=2)
        else:
            print("\nVector store is empty. Skipping example query.")

        print("\nRAG Pipeline processing complete!")

    except ValueError as ve:  # Catch specific error for API key
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}")
        import traceback

        traceback.print_exc()
