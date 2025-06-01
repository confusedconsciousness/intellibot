import os
import glob
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    PyPDFLoader
)
from langchain_core.documents import Document
from typing import List, Dict, Type


class DocumentManager:
    def __init__(self, source_directory: str):
        self.source_directory = source_directory
        self.supported_loaders: Dict[str, Type] = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".html": UnstructuredHTMLLoader,
            ".pdf": PyPDFLoader,
        }
        if not os.path.exists(self.source_directory):
            os.makedirs(self.source_directory)
            print(f"Created directory '{self.source_directory}'. Please add your documents there.")
            # Add sample files for first run
            self._create_sample_files()

    def _create_sample_files(self):
        """Creates sample files in the source directory for initial testing."""
        with open(os.path.join(self.source_directory, "sample.txt"), "w", encoding="utf-8") as f:
            f.write("This is a sample text file for testing the RAG pipeline.")
        with open(os.path.join(self.source_directory, "sample.md"), "w", encoding="utf-8") as f:
            f.write("# Sample Markdown\n\nThis is a test markdown document with some **bold** text.")
        with open(os.path.join(self.source_directory, "sample.html"), "w", encoding="utf-8") as f:
            f.write("<h1>Sample HTML</h1><p>This is a paragraph in an HTML document.</p>")
        print(f"Added sample files to '{self.source_directory}' for initial testing.")

    def load_documents(self) -> List[Document]:
        all_documents: List[Document] = []
        print(f"\n--- Loading Documents from '{self.source_directory}' ---")
        for ext, LoaderClass in self.supported_loaders.items():
            file_paths = glob.glob(os.path.join(self.source_directory, f"*{ext}"))
            for file_path in file_paths:
                try:
                    print(f"Loading {file_path}...")
                    loader = LoaderClass(file_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    print(f"Loaded {len(documents)} document(s) from {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        if not all_documents:
            print(f"No documents found or loaded from '{self.source_directory}'.")
        else:
            print(f"Total documents loaded: {len(all_documents)}")
        return all_documents
