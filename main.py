import streamlit as st
import logging
import os
import pdfplumber
import ollama
import warnings
import chromadb

from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader

from typing import List, Any

warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

PERSIST_DIRECTORY = os.path.join("data", "vectors")

st.set_page_config(
    page_title="Document Processing Agent",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def process_question(query_engine, question: str) -> str:
    print("Query engine is", query_engine)
    print("Question is ", question)
    response = query_engine.query(question)
    # print(response)
    return response

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    with pdfplumber.open(file_upload) as pdf:
        return [page.to_image().original for page in pdf.pages]



def main() -> None:
    st.subheader("Document Assistant", divider="gray", anchor=False)

    models_info = ollama.list()
    # available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "pdf_pages" not in st.session_state:
        st.session_state["pdf_pages"] = {}
    if "file_uploads" not in st.session_state:
        st.session_state["file_uploads"] = []
    if "index" not in st.session_state:
        st.session_state["index"] = None
    if "query_engine" not in st.session_state:
        st.session_state["query_engine"] = None
    if "files_processed" not in st.session_state:
        st.session_state["files_processed"] = set()

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.get_or_create_collection("npcilDocs")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    llm = Ollama(model="llama3.2", request_timeout=120.0)

    file_uploads = col1.file_uploader(
        "Upload one or more PDF files ↓",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if file_uploads:
        custom_temp_dir = './docs' 
        new_files_uploaded = False

        for file_upload in file_uploads:
            if file_upload.name not in st.session_state["files_processed"]:
                new_files_uploaded = True
                with st.spinner(f"Processing {file_upload.name}..."):
                    file_path = os.path.join(custom_temp_dir, file_upload.name)
                    file_upload.seek(0)  # Ensure we read from the start
                    with open(file_path, "wb") as f:
                        f.write(file_upload.read())

                    st.session_state["file_uploads"].append(file_upload)
                    with pdfplumber.open(file_upload) as pdf:
                        st.session_state["pdf_pages"][file_upload.name] = [
                            page.to_image().original for page in pdf.pages
                        ]

                    st.session_state["files_processed"].add(file_upload.name)

        if new_files_uploaded:
            # Build index only once for all uploaded files
            try:
                documents = SimpleDirectoryReader(custom_temp_dir, recursive=True).load_data()
                st.session_state["index"] = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    embed_model=embed_model,
                    show_progress=True,
                )
                st.session_state["query_engine"] = st.session_state["index"].as_query_engine(
                    llm=llm,
                    similarity_top_k=3,
                    streaming=False,
                )
            except Exception as e:
                st.error(f"Error building index: {e}")

    if st.session_state["pdf_pages"]:
        zoom_level = col1.slider("Zoom Level", 100, 1000, 700, 50, key="zoom_slider")
        with col1:
            for name, pages in st.session_state["pdf_pages"].items():
                st.markdown(f"**📄 {name}**")
                with st.container(height=410, border=True):
                    for page_image in pages:
                        st.image(page_image, width=zoom_level)

    with col2:
        message_container = st.container(height=500, border=True)

        # Display existing messages
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Prompt input box
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            # Add user message to history
            st.session_state["messages"].append({"role": "user", "content": prompt})

            # Display user message immediately
            with message_container.chat_message("user", avatar="😎"):
                st.markdown(prompt)

            # Process and display assistant response
            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(":green[processing...]"):
                    if st.session_state["pdf_pages"]:
                        response = process_question(st.session_state["query_engine"], prompt)
                        st.markdown(response)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
                    else:
                        st.warning("Please upload at least one PDF first.")

if __name__ == "__main__":
    main()
