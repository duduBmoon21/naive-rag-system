import streamlit as st
from helpers.loader import load_pdf
from helpers.chain import create_qa_chain
from helpers.vectorstore import VectorStore
from dotenv import load_dotenv
import os

# Config
load_dotenv()
st.set_page_config(page_title="PDF RAG (Groq)", layout="wide")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# UI
st.title("PDF Q&A with Groq (Llama 3)")
st.caption("Upload a PDF and ask questions - powered by Groq's ultra-fast LLMs")

# File upload
uploaded_file = st.file_uploader("Choose PDF", type="pdf")

if uploaded_file and not st.session_state.processed:
    with st.status("Processing PDF...", expanded=True) as status:
        try:
            # Save file
            os.makedirs("./data", exist_ok=True)
            temp_path = f"./data/{uploaded_file.name}"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and chunk
            st.write("Extracting text from PDF...")
            docs = load_pdf(temp_path)
            
            # Create vector store
            st.write("Generating embeddings...")
            st.session_state.vectorstore = VectorStore()
            st.session_state.vectorstore.create_from_documents(docs)
            
            st.session_state.processed = True
            status.update(label="Processing complete!", state="complete", expanded=False)
            st.success("Ready for questions!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Q&A Section
if st.session_state.processed:
    st.divider()
    question = st.text_input("Ask about the PDF:", placeholder="What's this document about?")
    
    if question:
        with st.spinner("Thinking..."):
            try:
                retriever = st.session_state.vectorstore.get_retriever()
                qa_chain = create_qa_chain(retriever)
                result = qa_chain({"query": question})
                
                # Display answer
                st.subheader("Answer")
                st.write(result["result"])
                
                # Show sources
                with st.expander("Source References"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.caption(f"Source {i+1} (Page {doc.metadata.get('page', '?')})")
                        st.text(doc.page_content[:500] + "...")
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")