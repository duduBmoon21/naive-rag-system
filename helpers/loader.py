from langchain.document_loaders import PyPDFLoader
from typing import List
from langchain.schema import Document

def load_pdf(file_path: str) -> List[Document]:
    """Load and return documents from a PDF file"""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        raise ValueError(f"Failed to load PDF: {str(e)}")