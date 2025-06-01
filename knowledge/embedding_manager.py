import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings


class EmbeddingManager:
    """
    Manages the initialization of the embedding model.
    """

    def __init__(self, model_name: str = "models/embedding-001"):
        self.model_name = model_name
        self.google_api_key = os.getenv("GEMINI_API_KEY")
        if not self.google_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it.")
        self.embeddings: Embeddings = self._initialize_embeddings()
        print(f"EmbeddingManager initialized with model: {model_name}")

    def _initialize_embeddings(self) -> Embeddings:
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(model=self.model_name, google_api_key=self.google_api_key)
            print("Google Generative AI Embeddings initialized successfully.")
            return embeddings_model
        except Exception as e:
            print(f"Error initializing Google embeddings: {e}")
            raise

    def get_embeddings(self) -> Embeddings:
        return self.embeddings
