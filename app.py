import os
import shutil
import streamlit as st
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_core.documents import Document
from helpers.loader import load_pdf, load_youtube_transcript
from helpers.chain import build_chain
from helpers.retriever import HybridRetriever

# ============================
# Page config
# ============================
st.set_page_config(page_title="Lumi - Your Study Assistant", layout="wide", page_icon="💡")

# ============================
# Utilities
# ============================
def safe_delete_folder(path, ignore_errors=True):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            if not ignore_errors:
                raise e
            print(f"Warning: Could not delete {path}. Continuing...")

# ============================
# Session state
# ============================
for key in ["collections", "active_collection", "messages", "last_preview", "sidebar_top_k", "sidebar_use_reranker"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key == "collections" else []

# ============================
# Header
# ============================
st.title("Lumi - Your Study Assistant")
st.caption("Upload study materials and get AI-powered insights")

# ============================
# Upload Section
# ============================
with st.expander("Upload Sources", expanded=True):
    collection_name = st.text_input("Collection Name", placeholder="Week1:Lecture Note")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader("PDF Documents", type="pdf", accept_multiple_files=True)
    with col2:
        youtube_url = st.text_input("YouTube Video URL", placeholder="https://youtube.com/watch?v=...")

process_btn = st.button("Process Materials")

if process_btn:
    if not collection_name:
        st.error("Provide a collection name.")
    elif collection_name in st.session_state.collections:
        st.error("Collection already exists.")
    elif not (uploaded_files or youtube_url):
        st.error("Upload PDF or YouTube URL.")
    else:
        safe_delete_folder("./data")
        os.makedirs("./data", exist_ok=True)
        retriever = HybridRetriever()
        all_docs = []

        # Process PDFs
        if uploaded_files:
            for f in uploaded_files:
                path = os.path.join("./data", f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                pdf_docs = load_pdf(path)
                all_docs.extend(pdf_docs)
                os.remove(path)

        # Process YouTube
        if youtube_url:
            yt_docs = load_youtube_transcript(youtube_url)
            all_docs.extend(yt_docs)

        if all_docs:
            retriever.ingest_documents(all_docs)
            st.session_state.collections[collection_name] = retriever
            st.session_state.active_collection = collection_name
            st.session_state.messages = []
            st.success(f"Processed {len(all_docs)} chunks! Ready for questions.")
        else:
            st.error("No valid content processed.")

# ============================
# Chat Section
# ============================
if st.session_state.active_collection:
    st.divider()
    st.subheader("Chat with Lumi")
    retriever_obj = st.session_state.collections[st.session_state.active_collection]

    # Get LC-compatible retriever
    active_retriever = retriever_obj.get_retriever(
        k=st.session_state.sidebar_top_k or 4,
        rerank=st.session_state.sidebar_use_reranker if "sidebar_use_reranker" in st.session_state else True
    )

    all_docs = retriever_obj.get_all_documents()

    # Chunk selection
    with st.expander("🔍 Select Specific Chunks (optional)", expanded=False):
        selected_chunk_indices = st.multiselect(
            "Chunks to focus on:",
            options=list(range(len(all_docs))),
            format_func=lambda i: f"Chunk {i+1}: {all_docs[i].page_content[:100]}..."
        )
        selected_chunks = [all_docs[i] for i in selected_chunk_indices] if selected_chunk_indices else None

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input question
    question = st.chat_input("Ask Lumi...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                qa_chain = build_chain(retriever=active_retriever)
                chat_history = st.session_state.messages[:-1]

                if selected_chunks:
                    # Manually selected chunks
                    result = retriever_obj.generate_answer_from_selected_chunks(
                        question=question,
                        chunks=selected_chunks,
                        chat_history=chat_history,
                        qa_chain_func=qa_chain
                    )
                else:
                    # Auto retrieval
                    result = qa_chain.invoke({"query": question, "chat_history": chat_history})


                # Identity check
                if result.get("identity_response"):
                    response = result.get("answer", "I'm Lumi, your assistant.")
                else:
                    response = f"""
**From Your Materials:**  
{result.get('context_answer', 'No information found.')}

**Lumi's Analysis:**  
{result.get('analysis_answer', 'No analysis generated.')}

"""
                placeholder.markdown(response)

                # Show sources
                if result.get("source_documents"):
                    with st.expander("View Source References", expanded=False):
                        for i, doc in enumerate(result["source_documents"]):
                            src_type = "🎬" if "youtube" in doc.metadata.get("type", "").lower() else "📄"
                            title = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
                            st.caption(f"{src_type} Source {i+1}: {title}")
                            st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                placeholder.markdown(f"Error: {str(e)}")

# ============================
# Reset
# ============================
if st.button("Start New Session"):
    safe_delete_folder("./data")
    for key in ["collections", "active_collection", "messages", "last_preview"]:
        st.session_state[key] = {} if key == "collections" else []
    st.rerun()


# =========================================================
# Sidebar Controls
# =========================================================
with st.sidebar:
    st.markdown("## Lumi Controls")
    st.markdown("---")
    st.markdown("### Document Collections")

    if not st.session_state.collections:
        st.info("No collections created yet. Upload materials to start.")
    else:
        def on_collection_change():
            st.session_state.messages = []

        st.selectbox(
            "Active Collection",
            options=list(st.session_state.collections.keys()),
            key="active_collection",
            on_change=on_collection_change,
            help="Switch between your processed document sets.",
        )

        st.markdown("---")
        if st.button("🗑️ Delete Current Collection"):
            collection_to_delete = st.session_state.active_collection
            if collection_to_delete:
                del st.session_state.collections[collection_to_delete]
                if st.session_state.collections:
                    st.session_state.active_collection = list(st.session_state.collections.keys())[0]
                else:
                    st.session_state.active_collection = None
                st.session_state.messages = []
                st.toast(f"Collection '{collection_to_delete}' deleted.", icon="✅")
                st.rerun()

    st.markdown("---")
    st.markdown("### Retrieval Settings")
    st.session_state.sidebar_top_k = st.number_input(
        "Top-K results (after rerank)",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="How many chunks to pass to the QA chain after hybrid retrieval + reranking?",
    )
    st.session_state.sidebar_use_reranker = st.checkbox(
        "Use reranker (Cross-Encoder)",
        value=True,
        help="Improves relevance by reordering candidate chunks before answering.",
    )

    st.markdown("---")
    st.markdown("### How to use Lumi")
    st.markdown(
        """
1. Name a new collection  
2. Upload PDFs or paste a YouTube link  
3. Click **Process Materials**  
4. Select the collection and **type** your question!
        """
    )
    st.markdown("---")
    st.markdown("**Tips for best results:**")
    st.markdown(
        """
- Use clear, specific questions  
- Combine multiple sources  
- Videos with English captions work best  
- Turn **reranker** on for better relevance (slightly slower)
        """
    )