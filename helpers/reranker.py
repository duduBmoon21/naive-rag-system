# helpers/reranker.py
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List

# Load a reranker model (smaller = faster, bigger = more accurate)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def rerank(query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
    """Re-rank a list of documents by semantic relevance to the query."""
    if not docs:
        return []
        
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(
        zip(docs, scores), key=lambda x: x[1], reverse=True
    )
    return [doc for doc, _ in scored_docs[:top_n]]
