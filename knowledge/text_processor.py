from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


class TextProcessor:
    """
    Handles splitting/chunking of documents.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        print(f"TextProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller chunks.
        """
        if not documents:
            print("No documents provided to split.")
            return []

        print("\n--- Chunking Documents ---")
        chunked_documents = self.splitter.split_documents(documents)
        print(f"Total chunks created: {len(chunked_documents)}")
        if not chunked_documents:
            print("No chunks were created. Check your documents and chunking parameters.")
        return chunked_documents