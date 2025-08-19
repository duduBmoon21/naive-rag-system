from typing import List, Optional, Dict
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseRetriever
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_core.runnables import Runnable, RunnableLambda
from .reranker import rerank as rerank_docs

class _HybridRetrieverWrapper(BaseRetriever):
    """Internal wrapper to make the hybrid function a LangChain retriever."""
    hybrid_retriever_func: Runnable

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.hybrid_retriever_func.invoke(query)


class HybridRetriever:
    """Hybrid dense + sparse in-memory retriever with optional reranking."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.bm25_index: Optional[BM25Okapi] = None
        self.doc_store: Dict[str, Document] = {}

    def ingest_documents(self, documents: List[Document]):
        """Split, index, and store documents."""
        if not documents:
            raise ValueError("No documents provided")

        split_docs = self.text_splitter.split_documents(documents)

        # Sparse BM25 index
        self._build_bm25_index(split_docs)

        # Dense in-memory vectorstore using FAISS
        self.vectorstore = FAISS.from_documents(
            documents=split_docs, embedding=self.embeddings
        )
  
    def get_all_documents(self) -> List[Document]:
        """Return all documents in the retriever."""
        return list(self.doc_store.values())

    def generate_answer_from_selected_chunks(self, question: str, chunks: List[Document], chat_history, qa_chain_func):
        """Generate answer from manually selected chunks."""
        from langchain.schema import BaseRetriever
        
        # Create a custom retriever that only returns the selected chunks
        class ManualRetriever(BaseRetriever):
            def get_relevant_documents(self, query: str) -> List[Document]:
                return chunks
        
        # Use the provided QA chain function with the manual retriever
        manual_retriever = ManualRetriever()
        qa_chain = qa_chain_func(manual_retriever)
        
        return qa_chain({"query": question, "chat_history": chat_history})

    def _build_bm25_index(self, documents: List[Document]):
        """Build BM25 index for sparse retrieval."""
        self.doc_store = {str(i): doc for i, doc in enumerate(documents)}
        corpus = [doc.page_content.split() for doc in documents]
        self.bm25_index = BM25Okapi(corpus)

    def get_hybrid_retriever_func(self, k: int = 4, rerank: bool = True):
        """Return a callable hybrid retriever function for in-app use."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")

        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k*2})

        if not rerank:
            return lambda query: base_retriever.get_relevant_documents(query)

        def hybrid_retriever(query: str) -> List[Document]:
            # Dense results
            dense_results = base_retriever.invoke(query)

            # Sparse BM25 results
            tokenized_query = query.split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            sparse_ids = np.argsort(bm25_scores)[-k*2:][::-1]
            sparse_results = [self.doc_store[str(i)] for i in sparse_ids]

            # Merge & deduplicate
            all_results = list({
                doc.page_content: doc
                for doc in dense_results + sparse_results
            }.values())

            # Re-rank final set
            return rerank_docs(query, all_results, top_n=k)

        return hybrid_retriever

    def get_retriever(self, k: int = 4, rerank: bool = True):
        """Return a BaseRetriever-compatible wrapper for RetrievalQA."""
        hybrid_func = self.get_hybrid_retriever_func(k=k, rerank=rerank)
        return _HybridRetrieverWrapper(
            hybrid_retriever_func=RunnableLambda(hybrid_func)
        )

    def clear(self):
        """Reset all indices."""
        self.vectorstore = None
        self.bm25_index = None
        self.doc_store = {}