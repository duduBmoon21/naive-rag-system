import os
import pypdf
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(file_path: str) -> List[Document]:
    """Load and validate PDF file with comprehensive error handling"""
    try:
        # 1. File validation
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError("PDF is empty (0 bytes)")
        
        # 2. PDF structure check
        with open(file_path, "rb") as f:
            pypdf.PdfReader(f) 
            
        # 3. Content extraction
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No readable text in PDF")
        
        # 4. Add metadata
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
        if os.path.exists(file_path):
            os.remove(file_path)  # Cleanup temp files
        raise ValueError(f"PDF processing failed: {str(e)}")