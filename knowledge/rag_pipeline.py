from dotenv import load_dotenv

from knowledge.document_manager import DocumentManager
from knowledge.text_processor import TextProcessor
from knowledge.embedding_manager import EmbeddingManager
from knowledge.vector_store_manager import VectorStoreManager


class RAGPipeline:
    """
    Orchestrates the RAG pipeline: loading, chunking, embedding, and storing documents.
    """

    def __init__(self, source_dir: str, chroma_dir: str, collection_name: str,
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 embedding_model_name: str = "models/embedding-001"):
        load_dotenv()  # Ensure API keys are loaded

        self.doc_manager = DocumentManager(source_directory=source_dir)
        self.text_processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        try:
            self.embedding_manager = EmbeddingManager(model_name=embedding_model_name)
            self.embeddings = self.embedding_manager.get_embeddings()
            self.vector_store_manager = VectorStoreManager(
                embedding_function=self.embeddings,
                db_directory=chroma_dir,
                collection_name=collection_name
            )
        except ValueError as e:  # Handles missing API key from EmbeddingManager
            print(f"Failed to initialize RAG Pipeline: {e}")
            raise  # Re-raise to stop execution if critical components fail
        except Exception as e:
            print(f"An unexpected error occurred during RAGPipeline initialization: {e}")
            raise

    def setup_vector_store(self, force_recreate: bool = False):
        """
        Loads documents, chunks them, and stores them in the vector store.
        If force_recreate is False, it will try to load an existing store first.
        """
        if not force_recreate:
            self.vector_store_manager.load_existing_store()
            if self.vector_store_manager.get_collection_count() > 0:
                print(
                    f"Vector store already exists and contains {self.vector_store_manager.get_collection_count()} items. Skipping document processing.")
                return

        print("\n--- Starting Document Processing and Vector Store Setup ---")
        # 1. Load documents
        raw_documents = self.doc_manager.load_documents()
        if not raw_documents:
            return

        # 2. Chunk documents
        chunked_documents = self.text_processor.split_documents(raw_documents)
        if not chunked_documents:
            return

        # 3. Store in VectorDB
        self.vector_store_manager.store_documents(chunked_documents)

    def query(self, query_text: str, k: int = 2):
        """
        Queries the vector store.
        """
        return self.vector_store_manager.query_documents(query_text, k=k)
