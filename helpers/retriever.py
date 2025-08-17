from typing import List, Optional, Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseRetriever
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
import os

class HybridRetriever:
    """Hybrid dense + sparse in-memory retriever"""

    def __init__(self, persist_dir: Optional[str] = None):
        # Force in-memory Chroma
        self.persist_dir = None
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.bm25_index: Optional[BM25Okapi] = None
        self.doc_store: Dict[str, Document] = {}
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def ingest_documents(self, documents: List[Document]):
        """Split, index, and store documents"""
        if not documents:
            raise ValueError("No documents provided")

        split_docs = self.text_splitter.split_documents(documents)

        # Sparse BM25 index
        self._build_bm25_index(split_docs)

        try:
            # Dense in-memory vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=None 
            )
        except Exception as e:
            if "sqlite3" in str(e).lower():
                print("Warning: Falling back to pure in-memory Chroma")
                # Try with explicit in-memory setting
                self.vectorstore = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                    persist_directory=None
                )
            else:
                raise

    def _build_bm25_index(self, documents: List[Document]):
        self.doc_store = {str(i): doc for i, doc in enumerate(documents)}
        corpus = [doc.page_content.split() for doc in documents]
        self.bm25_index = BM25Okapi(corpus)

    def get_hybrid_retriever_func(self, k: int = 4, rerank: bool = True):
        """Return a callable hybrid function for in-app use"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")

        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k*2})

        if not rerank:
            return lambda query: base_retriever.get_relevant_documents(query)

        def hybrid_retriever(query: str) -> List[Document]:
            dense_results = base_retriever.get_relevant_documents(query)
            tokenized_query = query.split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            sparse_ids = np.argsort(bm25_scores)[-k*2:][::-1]
            sparse_results = [self.doc_store[str(i)] for i in sparse_ids]

            # Combine and remove duplicates
            all_results = list({doc.page_content: doc for doc in dense_results + sparse_results}.values())
            return self._rerank(query, all_results)[:k]

        return hybrid_retriever

    def get_retriever(self, k: int = 4, rerank: bool = True):
        """Return a BaseRetriever-compatible wrapper for RetrievalQA"""
        hybrid_func = self.get_hybrid_retriever_func(k=k, rerank=rerank)

        class RetrieverWrapper(BaseRetriever):
            def get_relevant_documents(self, query: str) -> List[Document]:
                return hybrid_func(query)

        return RetrieverWrapper()

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank using CrossEncoder"""
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]

    def clear(self):
        """Reset all indices"""
        self.vectorstore = None
        self.bm25_index = None
        self.doc_store = {}