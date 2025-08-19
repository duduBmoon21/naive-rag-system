import sys
import streamlit as st
from helpers.loader import load_pdf
from helpers.youtube import load_youtube_transcript
from helpers.chain import create_qa_chain
from helpers.retriever import HybridRetriever
import os
import shutil

# --- Config ---
st.set_page_config(
    page_title="Lumi - Your Study Assistant",
    layout="wide",
    page_icon="💡"
)

# --- Cross-platform folder deletion ---
def safe_delete_folder(path, ignore_errors=True):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            if not ignore_errors:
                raise e
            print(f"Warning: Could not delete {path}. Continuing...")

# --- Session state ---
if "collections" not in st.session_state:
    st.session_state.collections = {} # Dict to store {name: retriever}
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "messages" not in st.session_state:
    st.session_state.messages = [] # Chat history for the active collection

# --- UI ---
st.title("Lumi - Your Study Assistant")
st.caption("Upload study materials and get AI-powered insights")

# --- File upload section ---
with st.expander("Upload Sources", expanded=True):
    collection_name = st.text_input(
        "Create a new collection name",
        placeholder="e.g., 'Week 1 Lecture Notes'",
        help="Give your document set a unique name."
    )
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "PDF Documents", type="pdf", accept_multiple_files=True,
            help="Upload lecture notes, research papers, or study materials"
        )
    with col2:
        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://youtube.com/watch?v=...",
            help="For best results, use videos with English captions"
        )

process_btn = st.button("Process Materials", type="primary")

# --- Process materials ---
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
                # Clear cache folders
                safe_delete_folder("./data")
                os.makedirs("./data", exist_ok=True)

                new_retriever = HybridRetriever()
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
                        st.error(f"{str(e)}")
                        st.video(youtube_url)

                # Create knowledge base
                if all_docs:
                    st.write(" Generating searchable knowledge...")
                    new_retriever.ingest_documents(all_docs)
                    st.session_state.collections[collection_name] = new_retriever
                    st.session_state.active_collection = collection_name
                    st.session_state.messages = [] # Clear chat for new collection
                    status.update(
                        label=f"Processed {processed_count} source(s)! Ready for questions.",
                        state="complete",
                        expanded=False
                    )
                    st.rerun()
                else:
                    st.error("No valid content could be processed")

            except Exception as e:
                st.error(f"Processing error: {str(e)}")

# --- Q&A Section ---
if st.session_state.active_collection:
    st.divider()
    st.subheader("Chat with Lumi")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if question := st.chat_input("What is the main idea of these documents?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Lumi is thinking..."):
                try:
                    active_retriever = st.session_state.collections[st.session_state.active_collection]
                    retriever = active_retriever.get_retriever()
                    qa_chain = create_qa_chain(retriever)
                    # Pass all but the last message for context
                    chat_history = st.session_state.messages[:-1]
                    result = qa_chain({"query": question, "chat_history": chat_history})

                    # Check for a direct identity response and format accordingly
                    if result.get("identity_response"):
                        response_content = result.get("answer", "I'm not sure how to answer that.")
                    else:
                        # Format the standard RAG response
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
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})

# --- Reset button ---
if st.button("Start New Session"):
    try:
        safe_delete_folder("./data")
        keys_to_clear = ["collections", "active_collection", "messages"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    except Exception as e:
        st.error(f"Reset failed: {str(e)}. Please restart the app.")

# --- Sidebar ---
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
            help="Switch between your processed document sets."
        )

        st.markdown("---")
        if st.button("🗑️ Delete Current Collection"):
            collection_to_delete = st.session_state.active_collection
            if collection_to_delete:
                # Remove the retriever object from the dictionary
                del st.session_state.collections[collection_to_delete]

                # Reset the active collection
                if st.session_state.collections:
                    # Set to the first available collection
                    st.session_state.active_collection = list(st.session_state.collections.keys())[0]
                else:
                    # No collections left
                    st.session_state.active_collection = None
                
                st.session_state.messages = [] # Clear chat history
                st.toast(f"Collection '{collection_to_delete}' deleted.", icon="✅")
                st.rerun()

    st.markdown("---")
    st.markdown("### How to use Lumi")
    st.markdown("""
    1. Name a new collection
    2. Upload PDFs or paste YouTube links
    3. Click **Process Materials**
    4. Select the collection and ask questions!
    """)
    st.markdown("---")
    st.markdown("**Tips for best results:**")
    st.markdown("""
    - Use clear, specific questions
    - Combine multiple sources
    - Videos with English captions work best
    """)