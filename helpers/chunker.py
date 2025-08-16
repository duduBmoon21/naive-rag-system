from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

def chunk_documents(documents: List[Document], 
                   chunk_size: int = 1000,
                   chunk_overlap: int = 200) -> List[Document]:
    """Split documents into chunks with overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)