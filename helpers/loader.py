import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import pypdf

def load_pdf(file_path: str) -> List[Document]:
    """Load PDF with robust error handling"""
    try:
        # 1. File existence
        if not os.path.exists(file_path):
            raise ValueError(f"File not found at {file_path}")
        
        # 2. File size check
        if os.path.getsize(file_path) == 0:
            raise ValueError("Uploaded PDF is empty (0 bytes)")
        
        # 3. PDF structure validation
        try:
            with open(file_path, "rb") as f:
                pypdf.PdfReader(f)  
        except pypdf.PdfReadError:
            raise ValueError("Invalid PDF structure (may be corrupted)")
        except Exception as e:
            raise ValueError(f"PDF validation failed: {str(e)}")
        
        # 4. Full content loading
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if not docs:
                raise ValueError("PDF contains no readable text")
            return docs
        except Exception as e:
            raise ValueError(f"Content extraction failed: {str(e)}")
            
    except Exception as e:
        # Cleanup temp files on failure
        if "temp_" in file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise