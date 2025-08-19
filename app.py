import os
import shutil
import tempfile

import streamlit as st

from typing import List
from langchain_core.documents import Document
from helpers.loader import load_pdf
from helpers.youtube import load_youtube_transcript
from helpers.chain import create_qa_chain
from helpers.retriever import HybridRetriever

# =========================================================
# Config
# =========================================================
st.set_page_config(
    page_title="Lumi - Your Study Assistant",
    layout="wide",
    page_icon="💡",
)

# =========================================================
# Utilities
# =========================================================
def safe_delete_folder(path, ignore_errors=True):
    """Cross-platform folder deletion with optional ignore."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            if not ignore_errors:
                raise e
            print(f"Warning: Could not delete {path}. Continuing...")

# =========================================================
# Session State
# =========================================================
if "collections" not in st.session_state:
    st.session_state.collections = {}
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_preview" not in st.session_state:
    st.session_state.last_preview = {}

# =========================================================
# Header
# =========================================================
st.title("Lumi - Your Study Assistant")
st.caption("Upload study materials and get AI-powered insights")

# =========================================================
# Upload Section
# =========================================================
with st.expander("Upload Sources", expanded=True):
    collection_name = st.text_input(
        "Create a new collection name",
        placeholder="e.g., 'Week 1 Lecture Notes'",
        help="Give your document set a unique name.",
    )
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "PDF Documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload lecture notes, research papers, or study materials",
        )
    with col2:
        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://youtube.com/watch?v=...",
            help="For best results, use videos with English captions",
        )

process_btn = st.button("Process Materials", type="primary")

# =========================================================
# Process Materials
# =========================================================
if process_btn:
    if not collection_name:
        st.error("Please provide a name for your collection.")
    elif collection_name in st.session_state.collections:
        st.error(f"A collection named '{collection_name}' already exists. Please choose a different name.")
    elif not (uploaded_files or youtube_url):
        st.error("Please upload at least one PDF or provide a YouTube URL.")
    else:
        with st.status(f"Processing materials for '{collection_name}'...", expanded=True) as status:
            try:
                # Reset temp data dir
                safe_delete_folder("./data")
                os.makedirs("./data", exist_ok=True)

                new_retriever = HybridRetriever()
                all_docs = []
                processed_count = 0

                # ----- Process PDFs -----
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        try:
                            st.write(f" Processing {uploaded_file.name[:50]} ...")
                            temp_path = os.path.join("./data", uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            # Load and split docs (your loader returns Documents)
                            pdf_docs = load_pdf(temp_path)
                            all_docs.extend(pdf_docs)
                            processed_count += 1

                            # ---------- Chunk Preview UI ----------
                            st.markdown(f"### Preview chunks from **{uploaded_file.name}**")
                            total_chunks = len(pdf_docs)
                            st.caption(f"📄 This PDF was split into **{total_chunks} chunks**.")
                            preview_mode = st.radio(
                                f"How do you want to preview **{uploaded_file.name}**?",
                                ["First N Chunks", "Pick Specific Chunk"],
                                key=f"preview_mode_{uploaded_file.name}",
                                horizontal=True,
                            )

                            if total_chunks > 0:
                                if preview_mode == "First N Chunks":
                                    n = st.number_input(
                                        f"How many chunks to preview for {uploaded_file.name}?",
                                        min_value=1,
                                        max_value=min(total_chunks, 20),
                                        value=min(5, total_chunks),
                                        key=f"first_n_{uploaded_file.name}",
                                    )
                                    for i, chunk in enumerate(pdf_docs[:n], start=1):
                                        st.caption(f"Chunk {i}:")
                                        st.text(chunk.page_content[:800])
                                        st.divider()
                                    # Save last preview meta
                                    st.session_state.last_preview[uploaded_file.name] = {
                                        "mode": "first_n",
                                        "count": int(n),
                                    }
                                else:
                                    selected_chunk = st.number_input(
                                        f"Enter chunk number for {uploaded_file.name}:",
                                        min_value=1,
                                        max_value=total_chunks,
                                        value=1,
                                        key=f"pick_chunk_{uploaded_file.name}",
                                    )
                                    st.caption(f"Chunk {selected_chunk}:")
                                    st.text(pdf_docs[selected_chunk - 1].page_content[:1200])
                                    st.divider()
                                    # Save last preview meta
                                    st.session_state.last_preview[uploaded_file.name] = {
                                        "mode": "single",
                                        "chunk_index": int(selected_chunk - 1),
                                    }
                            else:
                                st.warning("No chunks were created from this PDF.")
                            # --------------------------------------

                        except Exception as e:
                            st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                # ----- Process YouTube -----
                if youtube_url:
                    try:
                        st.write(" Processing YouTube video ...")
                        yt_docs = load_youtube_transcript(youtube_url)
                        all_docs.extend(yt_docs)
                        processed_count += 1
                    except Exception as e:
                        st.error(f"{str(e)}")
                        st.video(youtube_url)

                # ----- Build Knowledge Base -----
                if all_docs:
                    st.write(" Generating searchable knowledge ...")
                    new_retriever.ingest_documents(all_docs)
                    st.session_state.collections[collection_name] = new_retriever
                    st.session_state.active_collection = collection_name
                    st.session_state.messages = []  # reset chat for new collection
                    status.update(
                        label=f"Processed {processed_count} source(s)! Ready for questions.",
                        state="complete",
                        expanded=False,
                    )
                    st.rerun()
                else:
                    st.error("No valid content could be processed.")
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

# =========================================================
# Q&A Section 
# =========================================================
if st.session_state.active_collection:
    st.divider()
    st.subheader("Chat with Lumi")
    
    # Add chunk selection UI
    with st.expander("🔍 Select Specific Chunks to Use", expanded=False):
        active_retriever = st.session_state.collections[st.session_state.active_collection]
        all_docs = active_retriever.get_all_documents()
        
        st.write(f"Found {len(all_docs)} total chunks")
        
        # Let user select specific chunks
        selected_chunk_indices = st.multiselect(
            "Select chunks to focus on (optional):",
            options=list(range(len(all_docs))),
            format_func=lambda i: f"Chunk {i+1}: {all_docs[i].page_content[:100]}...",
            help="Select specific chunks to focus on. Leave empty to use automatic retrieval."
        )
        
        selected_chunks = [all_docs[i] for i in selected_chunk_indices] if selected_chunk_indices else None

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Text input
    question = st.chat_input("Type your question here...")

    if question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # --- Generate assistant response ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Lumi is thinking..."):
                try:
                    active_retriever = st.session_state.collections[st.session_state.active_collection]
                    
                    if selected_chunks:
                        # Use manually selected chunks
                        result = active_retriever.generate_answer_from_selected_chunks(
                            question=question,
                            chunks=selected_chunks,
                            chat_history=st.session_state.messages[:-1],
                            qa_chain_func=create_qa_chain  
                        )
                    else:
                        # Use automatic retrieval with settings
                        retriever = active_retriever.get_retriever(
                            k=st.session_state.sidebar_top_k,
                            rerank=st.session_state.sidebar_use_reranker
                        )
                        qa_chain = create_qa_chain(retriever) 
                        chat_history = st.session_state.messages[:-1]
                        result = qa_chain({"query": question, "chat_history": chat_history})
                    
                    if result.get("identity_response"):
                        response_content = result.get("answer", "I'm not sure how to answer that.")
                    else:
                         response_content = f"""
**From Your Materials:**  
{result.get('context_answer', 'No information found in sources.')}

**Lumi's Analysis:**  
{result.get('analysis_answer', 'No analysis was generated.')}
                        """
                    message_placeholder.markdown(response_content)

                    # Show sources
                    if result["source_documents"]:
                        with st.expander("View Source References", expanded=False):
                            for i, doc in enumerate(result["source_documents"]):
                                source_type = "🎬" if "youtube" in doc.metadata.get("type", "").lower() else "📄"
                                source_title = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
                                st.caption(f"{source_type} Source {i+1}: {source_title}")
                                st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                                st.divider()
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    response_content = "Sorry, I ran into an error. Please try again."
                    message_placeholder.markdown(response_content)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response_content})     
# =========================================================
# Reset Button
# =========================================================
if st.button("Start New Session"):
    try:
        safe_delete_folder("./data")
        for key in ["collections", "active_collection", "messages", "last_preview"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    except Exception as e:
        st.error(f"Reset failed: {str(e)}. Please restart the app.")

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