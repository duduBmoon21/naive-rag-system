import streamlit as st
from helpers.loader import load_pdf, load_youtube_transcript
from helpers.chain import create_qa_chain
from helpers.vectorstore import VectorStore
from dotenv import load_dotenv
import os
import shutil
import time

# Config
load_dotenv()
st.set_page_config(
    page_title="Lumi - Your Study Assistant", 
    layout="wide",
    page_icon="💡"
)

def safe_delete_folder(path, max_retries=3, delay=1):
    """Improved folder deletion with Windows workaround"""
    for i in range(max_retries):
        try:
            if os.path.exists(path):
                # Try renaming first to break locks
                temp_name = f"{path}_deleting_{i}"
                os.rename(path, temp_name)
                shutil.rmtree(temp_name)
                return True
        except Exception:
            if i == max_retries - 1:
                st.warning(f"Couldn't fully clear {path}. Some files may remain.")
                return False
            time.sleep(delay)
    return False

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "source_count" not in st.session_state:
    st.session_state.source_count = 0

# UI
st.title("Lumi - Your Study Assistant")
st.caption("Upload study materials and get AI-powered insights")

# File upload section
with st.expander("Upload Sources", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_files = st.file_uploader(
            "PDF Documents", 
            type="pdf",
            accept_multiple_files=True,
            help="Upload lecture notes, research papers, or study materials"
        )
    
    with col2:
        youtube_url = st.text_input(
            "YouTube Video URL", 
            placeholder="https://youtube.com/watch?v=...",
            help="For best results, use videos with English captions"
        )

process_btn = st.button("Process Materials", type="primary")

if process_btn and (uploaded_files or youtube_url):
    with st.status("Processing your materials...", expanded=True) as status:
        try:
            # Clear previous state
            if st.session_state.vectorstore:
                try:
                    st.session_state.vectorstore.close()
                except:
                    pass
            
            # Clear cache
            safe_delete_folder("./chroma_db")
            safe_delete_folder("./data")
            os.makedirs("./data", exist_ok=True)
            
            # Initialize new processing
            st.session_state.vectorstore = VectorStore()
            all_docs = []
            processed_count = 0
            
            # Process PDFs
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    try:
                        st.write(f" Processing {uploaded_file.name[:30]}...")
                        temp_path = f"./data/{uploaded_file.name}"
                        
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        pdf_docs = load_pdf(temp_path)
                        all_docs.extend(pdf_docs)
                        processed_count += 1
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            
            # Process YouTube
            if youtube_url:
                try:
                    st.write(" Processing YouTube video...")
                    yt_docs = load_youtube_transcript(youtube_url)
                    all_docs.extend(yt_docs)
                    processed_count += 1
                except Exception as e:
                    st.error(f" {str(e)}")
                    st.video(youtube_url)  
            
            # Create knowledge base if we got content
            if all_docs:
                st.write(" Generating searchable knowledge...")
                st.session_state.vectorstore.create_from_documents(all_docs)
                st.session_state.processed = True
                st.session_state.source_count = processed_count
                status.update(
                    label=f" Processed {processed_count} source(s)! Ready for questions.",
                    state="complete", 
                    expanded=False
                )
            else:
                st.error(" No valid content could be processed")
                st.session_state.processed = False
            
        except Exception as e:
            st.error(f" Processing error: {str(e)}")
            st.session_state.processed = False

# Q&A Section
if st.session_state.processed:
    st.divider()
    question = st.text_input(
        "Ask Lumi anything about your materials:",
        placeholder="What's the main idea of these documents?",
        help="Ask questions about concepts, summaries, or connections"
    )
    
    if question:
        with st.spinner(" Analyzing your question..."):
            try:
                retriever = st.session_state.vectorstore.get_retriever()
                qa_chain = create_qa_chain(retriever)
                result = qa_chain({"query": question})
                
                # Display answers
                with st.container():
                    st.subheader("From Your Materials")
                    st.markdown(result["context_answer"])
                    
                    st.subheader("Lumi's Analysis")
                    st.markdown(result["analysis_answer"])
                
                # Show sources
                if result["source_documents"]:
                    with st.expander("View Source References", expanded=False):
                        for i, doc in enumerate(result["source_documents"]):
                            source_type = "🎬" if "youtube" in doc.metadata.get("type", "").lower() else "📄"
                            source_title = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
                            st.caption(f"{source_type} Source {i+1}: {source_title}")
                            st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                            st.divider()
                
                if st.session_state.source_count == 1:
                    st.info("Tip: Add more sources for cross-referencing and richer insights")
                        
            except Exception as e:
                st.error(f" Error generating answer: {str(e)}")

# Reset button
if st.button("Start New Session"):
    try:
        if st.session_state.vectorstore:
            st.session_state.vectorstore.close()
        safe_delete_folder("./chroma_db")
        safe_delete_folder("./data")
        st.session_state.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Reset failed: {str(e)}. Please restart the app.")

# Sidebar info
with st.sidebar:
    st.markdown("## How to use Lumi")
    st.markdown("""
    1. Upload PDFs or paste YouTube links
    2. Click **Process Materials**
    3. Ask questions about the content
    4. Get AI-powered insights!
    """)
    st.markdown("---")
    st.markdown(" **Tips for best results:**")
    st.markdown("""
    - Use clear, specific questions
    - Combine multiple sources
    - Videos with English captions work best
    """)