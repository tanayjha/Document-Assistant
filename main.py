import streamlit as st
import logging
import os
import pdfplumber
import ollama
import warnings
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

# Suppress torch warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

st.set_page_config(
    page_title="📚 Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup logging
logging.basicConfig(level=logging.INFO)

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    with pdfplumber.open(file_upload) as pdf:
        return [page.to_image().original for page in pdf.pages]

# Main App
def main():
    st.markdown("## 🤖 Document Chat Assistant")

    # Session state
    for key, default in {
        "messages": [],
        "pdf_pages": {},
        "file_uploads": [],
        "index": None,
        "query_engine": None,
        "files_processed": set(),
        "memory_buffer": ChatMemoryBuffer.from_defaults(token_limit=2000),
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Sidebar
    with st.sidebar:
        st.markdown("### 📂 Uploaded Files")
        selected_doc = None
        if st.session_state["file_uploads"]:
            file_names = [f.name for f in st.session_state["file_uploads"]]
            selected_doc = st.selectbox("Choose a document to preview", file_names)
        else:
            st.info("Upload PDFs below to get started.")

        # Upload area
        uploaded_files = st.file_uploader(
            "📄 Upload PDF Files",
            type="pdf",
            accept_multiple_files=True
        )

    # Model and storage setup
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.get_or_create_collection("npcilDocs")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    llm = Ollama(model="llama3.2", request_timeout=120.0)

    # Process uploaded PDFs
    if uploaded_files:
        custom_temp_dir = "./docs"
        os.makedirs(custom_temp_dir, exist_ok=True)
        new_files = False

        for file_upload in uploaded_files:
            if file_upload.name not in st.session_state["files_processed"]:
                new_files = True
                with st.spinner(f"📥 Processing {file_upload.name}..."):
                    file_path = os.path.join(custom_temp_dir, file_upload.name)
                    with open(file_path, "wb") as f:
                        f.write(file_upload.read())
                    st.session_state["file_uploads"].append(file_upload)
                    st.session_state["pdf_pages"][file_upload.name] = extract_all_pages_as_images(file_upload)
                    st.session_state["files_processed"].add(file_upload.name)

        if new_files:
            try:
                with st.spinner("🔍 Building document index..."):
                    documents = SimpleDirectoryReader(custom_temp_dir, recursive=True).load_data()
                    index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=storage_context,
                        embed_model=embed_model,
                        show_progress=True,
                    )
                    st.session_state["index"] = index
                    st.session_state["query_engine"] = index.as_chat_engine(
                        llm=llm,
                        memory=st.session_state["memory_buffer"],
                        system_prompt="You are a helpful assistant that answers questions based on uploaded documents.",
                        chat_mode="context",
                        similarity_top_k=5
                    )
                    st.success("✅ Index built successfully!")
            except Exception as e:
                st.error(f"❌ Error building index: {e}")

    # Tabs for Chat + PDF Viewer
    chat_tab, pdf_tab = st.tabs(["💬 Chat", "📄 Document Viewer"])

    # PDF Viewer Tab
    with pdf_tab:
        if selected_doc and selected_doc in st.session_state["pdf_pages"]:
            st.markdown(f"### 📑 Preview: {selected_doc}")
            zoom = st.slider("🔍 Zoom", 100, 1000, 600, 50)
            cols = st.columns(3)
            for i, page in enumerate(st.session_state["pdf_pages"][selected_doc]):
                with cols[i % 3]:
                    st.image(page, caption=f"Page {i+1}", width=zoom)
        else:
            st.info("Select a document from the sidebar to preview.")

    # Chat Tab
    with chat_tab:
        chat_container = st.container(height=500, border=True)

        # Display messages
        for message in st.session_state["messages"]:
            avatar = "😎" if message["role"] == "user" else "🤖"
            with chat_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Prompt input
        user_input = st.chat_input("Ask something about your documents...", key="chat_input")

        if user_input:
            st.session_state["pending_prompt"] = user_input

        if "pending_prompt" in st.session_state:
            prompt = st.session_state["pending_prompt"]
            st.session_state["messages"].append({"role": "user", "content": prompt})

            with chat_container.chat_message("user", avatar="😎"):
                st.markdown(prompt)

            with chat_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        start = time.time()
                        response = st.session_state["query_engine"].chat(prompt)
                        end = time.time()

                        st.markdown(response.response)
                        st.caption(f"🕒 Response time: {end - start:.2f} seconds")

                        with st.expander("📚 Sources"):
                            for node in response.source_nodes:
                                file = node.metadata.get("file_name", "Unknown")
                                text = node.node.text.strip().replace("\n", " ")[:300]
                                st.markdown(f"**{file}** — `{text}...`")

                        st.session_state["messages"].append(
                            {"role": "assistant", "content": response.response}
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            del st.session_state["pending_prompt"]

        # Download chat history
        if st.session_state["messages"]:
            chat_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state["messages"]])
            st.download_button("📥 Download Chat History", chat_text, file_name="chat_history.txt")

if __name__ == "__main__":
    main()