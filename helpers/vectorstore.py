from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict

class VectorStore:
    """VectorStore using Chroma in-memory backend (no SQLite needed)"""
    
    def __init__(self):
        self.collections: Dict[str, Chroma] = {}
        self.current_collection: Optional[str] = None
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception as e:
            raise RuntimeError(f"Embedding model failed to load: {str(e)}")

    def create_collection(self, collection_name: str, documents: List[Document]):
        if not documents:
            raise ValueError("No documents provided for collection")
        
        # Create in-memory Chroma collection
        self.collections[collection_name] = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=None, 
            collection_name=collection_name
        )
        self.current_collection = collection_name
        return self.collections[collection_name]

    def switch_collection(self, collection_name: str):
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        self.current_collection = collection_name
        return self.collections[collection_name]

    def get_active_collection(self) -> Chroma:
        if not self.current_collection:
            raise ValueError("No active collection selected")
        return self.collections[self.current_collection]

    def create_from_documents(self, documents: List[Document]):
        return self.create_collection("default", documents)

    def get_retriever(self, k: int = 3):
        return self.get_active_collection().as_retriever(search_kwargs={"k": k})

    def close(self):
        """Reset all collections (memory-only, no file deletion needed)"""
        self.collections = {}
        self.current_collection = None
