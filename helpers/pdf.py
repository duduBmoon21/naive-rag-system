import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import pypdf

def load_pdf(file_path: str) -> List[Document]:
    """Load PDF safely and add metadata"""
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError("PDF is empty (0 bytes)")
    
    try:
        # Quick structure check
        with open(file_path, "rb") as f:
            pypdf.PdfReader(f)
        
        # Load content
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No readable text in PDF")

        # Add metadata
        for doc in docs:
            doc.metadata.update({
                "source": os.path.basename(file_path),
                "type": "pdf",
                "pages": len(docs)
            })
        return docs

    except pypdf.PdfReadError:
        raise ValueError("Corrupted or invalid PDF file")
    except Exception as e:
        raise ValueError(f"PDF processing failed: {str(e)}")
