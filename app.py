import streamlit as st
from helpers.loader import load_pdf
from helpers.chunker import chunk_documents

st.title("PDF RAG System")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save temporarily
        temp_path = f"./data/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and chunk
        docs = load_pdf(temp_path)
        chunks = chunk_documents(docs)
        
        st.success(f"Loaded {len(docs)} pages, split into {len(chunks)} chunks")
        st.json(chunks[0].metadata)  # Show sample metadata