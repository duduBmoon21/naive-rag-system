from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict
import os

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", cache_folder: str = "./embedding_cache"):
        """
        Enhanced vector store with multi-collection support
        """
        self.persist_directory = persist_directory
        self.collections: Dict[str, Chroma] = {}
        self.current_collection: Optional[str] = None
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder=cache_folder
            )
        except Exception as e:
            raise RuntimeError(f"Embedding model failed to load: {str(e)}")

    def create_collection(self, collection_name: str, documents: List[Document]):
        """
        Create a new named collection from documents
        """
        if not documents:
            raise ValueError("No documents provided for collection")
            
        try:
            collection_path = os.path.join(self.persist_directory, collection_name)
            os.makedirs(collection_path, exist_ok=True)
            
            self.collections[collection_name] = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=collection_path,
                collection_name=collection_name
            )
            self.current_collection = collection_name
            return self.collections[collection_name]
            
        except Exception as e:
            if "Lock" in str(e):
                raise RuntimeError(f"Collection {collection_name} is locked - try deleting its folder")
            raise RuntimeError(f"Collection creation failed: {str(e)}")

    def switch_collection(self, collection_name: str):
        """
        Switch active collection
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        self.current_collection = collection_name
        return self.collections[collection_name]

    def get_active_collection(self) -> Chroma:
        """
        Get current active collection
        """
        if not self.current_collection:
            raise ValueError("No active collection selected")
        return self.collections[self.current_collection]

    # Legacy methods (maintain backward compatibility)
    def create_from_documents(self, documents: List[Document]):
        """Legacy: Creates default collection"""
        return self.create_collection("default", documents)

    def load_existing(self, collection_name: str = "default"):
        """Legacy: Loads specified collection"""
        collection_path = os.path.join(self.persist_directory, collection_name)
        if not os.path.exists(collection_path):
            raise FileNotFoundError(f"No collection found at {collection_path}")

        self.collections[collection_name] = Chroma(
            persist_directory=collection_path,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        self.current_collection = collection_name
        return self.collections[collection_name]

    def get_retriever(self, k: int = 3):
        """Get retriever from active collection"""
        return self.get_active_collection().as_retriever(search_kwargs={"k": k})

    def close(self):
        """Clean up all collections"""
        try:
            for collection in self.collections.values():
                collection.delete_collection()
            self.collections = {}
            self.current_collection = None
        except Exception as e:
            print(f"Warning during cleanup: {e}")