from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class HybridRetriever:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_dir = persist_dir
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def ingest_documents(self, documents: List[Document]):
        """Process and store documents in vectorstore"""
        if not documents:
            raise ValueError("No documents provided")
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Create or update vectorstore
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.vectorstore.add_documents(split_docs)

    def get_retriever(self, k: int = 4):
        """Return configured retriever"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def clear(self):
        """Reset the retriever"""
        if self.vectorstore:
            self.vectorstore.delete_collection()
        self.vectorstore = None