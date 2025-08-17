from typing import List, Optional, Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

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
        self.bm25_index: Optional[BM25Okapi] = None
        self.doc_store: Dict[str, Document] = {}  # {doc_id: Document}
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def ingest_documents(self, documents: List[Document]):
        """Process and store documents with both dense and sparse indexing"""
        if not documents:
            raise ValueError("No documents provided")
        
        # Split and store documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Create BM25 sparse index
        self._build_bm25_index(split_docs)
        
        # Create/update Chroma vectorstore
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.vectorstore.add_documents(split_docs)

    def _build_bm25_index(self, documents: List[Document]):
        """Build sparse retrieval index"""
        self.doc_store = {str(i): doc for i, doc in enumerate(documents)}
        corpus = [doc.page_content.split() for doc in documents]
        self.bm25_index = BM25Okapi(corpus)

    def get_retriever(self, k: int = 4, rerank: bool = True):
        """Return retriever with optional hybrid reranking"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
            
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k*2})  # Fetch extra for reranking
        
        if not rerank:
            return base_retriever
            
        def hybrid_retriever(query: str) -> List[Document]:
            # Dense retrieval
            dense_results = base_retriever.get_relevant_documents(query)
            
            # Sparse retrieval (BM25)
            tokenized_query = query.split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            sparse_ids = np.argsort(bm25_scores)[-k*2:][::-1]
            sparse_results = [self.doc_store[str(i)] for i in sparse_ids]
            
            # Combine and rerank
            all_results = list(set(dense_results + sparse_results))
            return self._rerank(query, all_results)[:k]
            
        return hybrid_retriever

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Re-rank results using cross-encoder"""
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]

    def clear(self):
        """Reset all retrieval indices"""
        if self.vectorstore:
            self.vectorstore.delete_collection()
        self.vectorstore = None
        self.bm25_index = None
        self.doc_store = {}