from typing import List
from langchain_core.documents import Document
from .pdf import load_pdf as _load_pdf
from .youtube import load_youtube_transcript as _load_youtube

def load_pdf(file_path: str) -> List[Document]:
    """Public interface for PDF loading (cross-platform safe)"""
    try:
        return _load_pdf(file_path)
    except Exception as e:
        raise ValueError(f"PDF Error: {str(e)}")

def load_youtube_transcript(url: str) -> List[Document]:
    """Public interface for YouTube loading (cross-platform safe)"""
    try:
        return _load_youtube(url)
    except Exception as e:
        raise ValueError(f"YouTube Error: {str(e)}")
