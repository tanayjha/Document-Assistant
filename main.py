import os
import warnings

# Suppress all warnings first
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set environment variables before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Configure torch threading BEFORE any torch imports
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except (ImportError, RuntimeError):
    pass

# Set multiprocessing start method before other imports
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import streamlit as st
import logging
import pdfplumber
import ollama
import chromadb
import time

from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer

from typing import List, Any

st.set_page_config(
    page_title="📚 Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Constants
TEMP_DIR = "./docs"
MODEL_CACHE_DIR = "./model_cache"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
OLLAMA_MODEL = "llama3.2"
SYSTEM_PROMPT = """You are a precise technical assistant that answers questions based ONLY on the provided documents.

CRITICAL RULES:
1. Answer ONLY using information explicitly stated in the documents
2. If the answer is not in the documents, explicitly state: "This information is not available in the provided documents"
3. Be specific and cite which document sections support your answer
4. For specification documents, prioritize accuracy over being helpful - it's better to say you don't know than to guess
5. If the question requires information from multiple documents, clearly indicate which document covers which part
6. Do not make assumptions or infer beyond what is explicitly stated in the documents

Always be honest about the limitations of the available information."""
SIMILARITY_TOP_K = 5
MEMORY_TOKEN_LIMIT = 2000

@st.cache_data
def extract_all_pages_as_images(file_path: str) -> List[Any]:
    """Extract all pages from a PDF file as images."""
    with pdfplumber.open(file_path) as pdf:
        return [page.to_image().original for page in pdf.pages]

def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "pdf_pages": {},
        "file_uploads": [],
        "index": None,
        "query_engine": None,
        "files_processed": set(),
        "memory_buffer": ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT),
        "chroma_client": None,
        "chroma_collection": None,
        "vector_store": None,
        "storage_context": None,
        "embed_model": None,
        "llm": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def create_query_engine():
    """Create a new query engine with current settings."""
    if st.session_state["index"] is None or st.session_state["llm"] is None:
        return None

    return st.session_state["index"].as_chat_engine(
        llm=st.session_state["llm"],
        memory=st.session_state["memory_buffer"],
        system_prompt=SYSTEM_PROMPT,
        chat_mode="context",
        similarity_top_k=SIMILARITY_TOP_K
    )

# Main App
def main():
    # Beautiful header with gradient
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 2rem 0;">
        <h1 style="margin: 0; font-size: 3rem;">🤖 Document Chat Assistant</h1>
        <p style="color: #cbd5e1; font-size: 1.1rem; margin-top: 0.5rem;">Ask questions about your documents with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h3 style="color: #3b82f6; margin-top: 0;">📂 Uploaded Documents</h3>
        </div>
        """, unsafe_allow_html=True)

        selected_doc = None
        if st.session_state["file_uploads"]:
            file_names = [f.name for f in st.session_state["file_uploads"]]
            st.markdown(f"<p style='color: #cbd5e1; font-size: 0.9rem;'>{len(file_names)} document(s) loaded</p>", unsafe_allow_html=True)
            selected_doc = st.selectbox("Choose a document to preview", file_names, key="doc_select")
        else:
            st.markdown("""
            <div style="padding: 1rem; background-color: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6; border-radius: 6px;">
                <p style="color: #cbd5e1; margin: 0; font-size: 0.95rem;">📄 No documents uploaded yet.<br>Start by uploading PDF files below.</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Upload area with better styling
        st.markdown("<h4 style='color: #a78bfa; margin-top: 1.5rem; margin-bottom: 0.5rem;'>📥 Upload PDF Files</h4>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drag and drop PDFs here or click to browse",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )

    # Process uploaded PDFs
    if uploaded_files:
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        new_files = False

        for file_upload in uploaded_files:
            if file_upload.name not in st.session_state["files_processed"]:
                new_files = True
                with st.spinner(f"📥 Processing {file_upload.name}..."):
                    file_path = os.path.join(TEMP_DIR, file_upload.name)
                    with open(file_path, "wb") as f:
                        f.write(file_upload.read())
                    st.session_state["file_uploads"].append(file_upload)
                    st.session_state["pdf_pages"][file_upload.name] = extract_all_pages_as_images(file_path)
                    st.session_state["files_processed"].add(file_upload.name)

        if new_files:
            try:
                # Initialize all components only when files are uploaded (first time only)
                if st.session_state["chroma_client"] is None:
                    with st.spinner("🔄 Initializing database..."):
                        st.session_state["chroma_client"] = chromadb.EphemeralClient()
                        st.session_state["chroma_collection"] = st.session_state["chroma_client"].get_or_create_collection("npcilDocs")
                        st.session_state["vector_store"] = ChromaVectorStore(chroma_collection=st.session_state["chroma_collection"])
                        st.session_state["storage_context"] = StorageContext.from_defaults(vector_store=st.session_state["vector_store"])

                if st.session_state["embed_model"] is None:
                    with st.spinner("🔄 Loading embedding model (this may take a minute)..."):
                        st.session_state["embed_model"] = HuggingFaceEmbedding(
                            model_name=EMBEDDING_MODEL,
                            device="cpu",
                            cache_folder=MODEL_CACHE_DIR
                        )

                if st.session_state["llm"] is None:
                    with st.spinner("🔄 Initializing LLM..."):
                        st.session_state["llm"] = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)

                with st.spinner("🔍 Building document index..."):
                    documents = SimpleDirectoryReader(TEMP_DIR, recursive=True).load_data()
                    index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=st.session_state["storage_context"],
                        embed_model=st.session_state["embed_model"],
                        show_progress=True,
                    )
                    st.session_state["index"] = index
                    st.session_state["query_engine"] = create_query_engine()
                    st.success("✅ Index built successfully!")
            except Exception as e:
                st.error(f"❌ Error building index: {e}")
                logging.error(f"Index building error: {e}", exc_info=True)

    # Tabs for Chat + PDF Viewer
    chat_tab, pdf_tab = st.tabs(["💬 Chat", "📄 Document Viewer"])

    # PDF Viewer Tab
    with pdf_tab:
        if selected_doc and selected_doc in st.session_state["pdf_pages"]:
            st.markdown(f"<h3 style='color: #3b82f6;'>📑 Document Preview: <span style='color: #a78bfa;'>{selected_doc}</span></h3>", unsafe_allow_html=True)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("<small style='color: #cbd5e1;'>Total pages: <strong style='color: #3b82f6;'>{}</strong></small>".format(len(st.session_state["pdf_pages"][selected_doc])), unsafe_allow_html=True)
            with col2:
                zoom = st.slider("🔍 Zoom", 100, 1000, 600, 50, label_visibility="collapsed")

            cols = st.columns(3)
            for i, page in enumerate(st.session_state["pdf_pages"][selected_doc]):
                with cols[i % 3]:
                    st.markdown(f"<p style='text-align: center; color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0.5rem;'><strong>Page {i+1}</strong></p>", unsafe_allow_html=True)
                    st.image(page, width=zoom, use_container_width=False)
        else:
            st.markdown("""
            <div style="padding: 2rem; text-align: center; background-color: rgba(59, 130, 246, 0.05); border: 1px dashed #3b82f6; border-radius: 8px;">
                <p style="color: #cbd5e1; font-size: 1.1rem; margin: 0;">📄 Select a document from the sidebar to preview</p>
            </div>
            """, unsafe_allow_html=True)

    # Chat Tab
    with chat_tab:
        st.markdown("<h3 style='color: #3b82f6; margin-bottom: 1.5rem;'>💬 Conversation</h3>", unsafe_allow_html=True)

        # Display messages
        for message in st.session_state["messages"]:
            avatar = "😎" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Prompt input
        user_input = st.chat_input("Ask something about your documents...", key="chat_input")

        if user_input:
            # Check if query engine is available
            if st.session_state["query_engine"] is None:
                st.warning("⚠️ Please upload documents first before asking questions.")
            else:
                st.session_state["messages"].append({"role": "user", "content": user_input})

                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🤖 Thinking..."):
                        try:
                            start = time.time()
                            response = st.session_state["query_engine"].chat(user_input)
                            end = time.time()

                            # Display response with enhanced styling
                            st.markdown(f"""
                            <div style="line-height: 1.8; color: #f1f5f9;">
                            {response.response}
                            </div>
                            """, unsafe_allow_html=True)

                            # Metrics with better styling
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"<small style='color: #94a3b8;'>⏱️ Response time: <span style='color: #3b82f6;'>{end - start:.2f}s</span></small>", unsafe_allow_html=True)

                            source_count = len(response.source_nodes)
                            with col2:
                                st.markdown(f"<small style='color: #94a3b8;'>📚 From <span style='color: #3b82f6;'>{source_count}</span> section(s)</small>", unsafe_allow_html=True)

                            st.divider()

                            # Enhanced sources display
                            with st.expander("📚 View Sources & References", expanded=False):
                                if response.source_nodes:
                                    for idx, node in enumerate(response.source_nodes, 1):
                                        file = node.metadata.get("file_name", "Unknown")
                                        page = node.metadata.get("page_label", node.metadata.get("page_number", "N/A"))
                                        text = node.node.text.strip().replace("\n", " ")[:300]

                                        st.markdown(f"""
                                        <div style="padding: 1rem; background-color: rgba(59, 130, 246, 0.08); border-left: 3px solid #3b82f6; border-radius: 6px; margin-bottom: 0.8rem;">
                                            <p style="margin: 0 0 0.5rem 0; color: #3b82f6; font-weight: 600;">
                                                📄 Source {idx}: {file}
                                            </p>
                                            <p style="margin: 0 0 0.5rem 0; color: #cbd5e1; font-size: 0.9rem;">
                                                📖 Page: <strong>{page}</strong>
                                            </p>
                                            <p style="margin: 0; color: #cbd5e1; font-size: 0.9rem; font-style: italic;">
                                                "{text}..."
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No sources available for this response.")

                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response.response}
                            )
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": error_msg}
                            )
                            logging.error(f"Query error: {e}", exc_info=True)

                st.rerun()

        # Chat controls
        if st.session_state["messages"]:
            st.divider()
            st.markdown("<h4 style='color: #a78bfa; margin-bottom: 1rem;'>🛠️ Chat Actions</h4>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                chat_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state["messages"]])
                st.download_button("📥 Download Chat History", chat_text, file_name="chat_history.txt", key="download_chat")
            with col2:
                if st.button("🗑️ Clear Chat History", key="clear_chat"):
                    st.session_state["messages"] = []
                    st.session_state["memory_buffer"] = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT)
                    st.session_state["query_engine"] = create_query_engine()
                    st.rerun()

if __name__ == "__main__":
    main()