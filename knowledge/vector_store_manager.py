from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional


class VectorStoreManager:
    def __init__(self, embedding_function: Embeddings, db_directory: str, collection_name: str):
        self.embedding_function = embedding_function
        self.db_directory = db_directory
        self.collection_name = collection_name
        self.db: Optional[Chroma] = None
        print(f"💾 VectorStoreManager initialized for directory '{db_directory}' and collection '{collection_name}'.")

    def store_documents(self, documents: List[Document]):
        if not documents:
            print("No documents provided to store.")
            return

        print("\n--- Storing Chunks in ChromaDB ---")
        try:
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                collection_name=self.collection_name,
                persist_directory=self.db_directory
            )
            print(
                f"Chunks embedded and stored in ChromaDB collection '{self.collection_name}' at '{self.db_directory}'.")
            print(f"Number of items in collection: {self.get_collection_count()}")
        except Exception as e:
            print(f"Error storing documents in Chroma: {e}")
            raise

    def load_existing_store(self):
        """Loads an existing ChromaDB store if it exists."""
        try:
            self.db = Chroma(
                persist_directory=self.db_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )
            print(
                f"Successfully loaded existing ChromaDB from '{self.db_directory}' for collection '{self.collection_name}'.")
            print(f"Number of items in collection: {self.get_collection_count()}")
        except Exception as e:  # Broad exception, Chroma can raise various errors if dir/collection doesn't exist as expected
            print(
                f"Could not load existing ChromaDB from '{self.db_directory}' for collection '{self.collection_name}'. A new one may be created on store_documents. Error: {e}")
            self.db = None

    def query_documents(self, query_text: str, k: int = 2) -> Optional[List[Document]]:
        """
        Queries the vector store for similar documents.
        """
        if not self.db:
            print("Vector store not initialized or loaded. Cannot query.")
            self.load_existing_store()  # Attempt to load before failing
            if not self.db:
                print("Still unable to query. Please store documents first.")
                return None

        if self.get_collection_count() == 0:
            print("Collection is empty. Cannot query.")
            return None

        print(f"\n--- Querying Collection ---")
        try:
            retrieved_docs = self.db.similarity_search(query_text, k=k)
            if retrieved_docs:
                print(f"Results for query: '{query_text}' (top {k})")
                for i, doc in enumerate(retrieved_docs):
                    print(f"\n--- Result {i + 1} ---")
                    print(f"Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"Content snippet: {doc.page_content[:10]}...")
            else:
                print("No results found for the query.")
            return retrieved_docs
        except Exception as e:
            print(f"Error during query: {e}")
            return None

    def get_collection_count(self) -> int:
        """
        Returns the number of items in the collection.
        """
        if self.db and self.db._collection:
            return self.db._collection.count()
        return 0
