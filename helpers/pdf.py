import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import pypdf

def load_pdf(file_path: str) -> List[Document]:
    """Load PDF safely and add metadata (cross-platform & in-memory compatible)"""
    
    # 1. Validate file existence
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError("PDF is empty (0 bytes)")
    
    try:
        # 2. Quick structure check to ensure PDF is readable
        with open(file_path, "rb") as f:
            pypdf.PdfReader(f)
        
        # 3. Load content using LangChain PDF loader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No readable text in PDF")

        # 4. Add metadata for each document chunk
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
        # No platform-specific cleanup here; fully cross-platform
        raise ValueError(f"PDF processing failed: {str(e)}")
