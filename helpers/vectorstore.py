from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Optional
import os

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", cache_folder: str = "./embedding_cache"):
        """
        Initialize embeddings and prepare vector store instance with updated imports.
        """
        self.persist_directory = persist_directory
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder=cache_folder
            )
            self.db: Optional[Chroma] = None
        except Exception as e:
            raise RuntimeError(f"Embedding model failed to load: {str(e)}")

    def create_from_documents(self, documents: List[Document]):
        """
        Create and persist vector store from a list of documents.
        """
        if not documents:
            raise ValueError("No documents provided for vector store")

        try:
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            return self.db
        except Exception as e:
            if "Lock" in str(e):
                raise RuntimeError("Database locked - try deleting the chroma_db folder")
            raise RuntimeError(f"Vector store creation failed: {str(e)}")

    def close(self):
        """Properly clean up resources"""
        try:
            if self.db:
                # ChromaDB doesn't have explicit close, but we can delete reference
                self.db = None
        except Exception as e:
            print(f"Warning: Error closing vectorstore - {e}")

    def load_existing(self):
        """
        Load an already persisted vector store (if exists).
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"No existing vector store found at {self.persist_directory}")

        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return self.db
        except Exception as e:
            raise RuntimeError(f"Failed to load existing vector store: {str(e)}")

    def get_retriever(self, k: int = 3):
        """
        Return retriever for similarity search.
        """
        if not self.db:
            raise ValueError("Vector store not initialized. Create or load it first.")
        return self.db.as_retriever(search_kwargs={"k": k})