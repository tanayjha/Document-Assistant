## Step 1: Install python

Run the following to see if one of them produces an output to confirm if python is already installed:
`python --version` or `python3 --version`

If they say command not found, install python using the following steps:
(Any 3.10+ version should work but I have tested using 3.10.7 hence installing that)

1. Download the macOS installer for Python 3.10.7: https://www.python.org/ftp/python/3.10.7/python-3.10.7-macos11.pkg
2. Run the installer and follow the prompts.
3. Verify installation: `python3 --version`

### Step 2: Install Ollama

First, download and install Ollama from the official website: https://ollama.com/download/.

Once the models are installed, run the following command in terminal:

`ollama run llama3.2`

## Step 3: Clone the code

## Step 4: Create virtual environment

### On Linux/Mac:
```
python3.11 -m venv myvenv
source myvenv/bin/activate
```

### On Windows:

```
python3.11 -m venv myvenv
myvenv\Scripts\activate
```

### Step 5: Install dependencies

`pip install -r requirements.txt`

### Step 6: Run the code

From the main folder run the following command:
`streamlit run main.py`

This should open up the app in: http://localhost:8501/

Here is a writeup for this project

## Abstract

This project explores the development of a fully local Retrieval-Augmented Generation (RAG) pipeline using large language models (LLMs) to enable private, document-based question answering. The motivation behind this work stems from growing concerns around data privacy, cloud dependency, and the need for user control in document parsing workflows. Existing cloud-based AI solutions typically require users to upload potentially sensitive documents to external servers and often impose restrictions on file size or usage, especially in free tiers. Moreover, fine-grained control over how documents are parsed and indexed can be critical for achieving more accurate results—something often difficult to customize in off-the-shelf cloud platforms.

To address these concerns, we implemented a local solution using the Ollama framework to run LLMs natively. The RAG architecture was built using ollama-index for orchestrating document ingestion and retrieval, ChromaDB as the vector store, and local embedding models for vectorization.

The system supports the following core functionalities:

1. An interactive user interface (built with Streamlit) for uploading documents and querying them via a chatbot.
2. A local RAG pipeline that indexes uploaded documents and retrieves relevant chunks based on user queries.
3. An entirely offline setup, ensuring privacy and independence from cloud services.

Key outcomes include a hands-on understanding of RAG workflows, practical experience with deploying LLMs locally (e.g., using LLaMA 3.2), and the creation of a personal document assistant capable of parsing, indexing, and responding to user questions—entirely within a secure local environment.

## Technology Stack

To implement a fully local Retrieval-Augmented Generation (RAG) system with an interactive interface, we relied on a suite of Python-based open-source tools. The components were chosen for their flexibility, community support, and ability to run without an internet connection. Below is a detailed breakdown of the tools and libraries used:

### User Interface
Streamlit (streamlit==1.40.0): A lightweight Python library for building web applications. It powers the front end, allowing users to upload documents and interact with the system through a simple chatbot-style interface.

### Document Ingestion and Parsing
pdfplumber (pdfplumber==0.11.4): Used to extract clean and structured text from uploaded PDF documents. Its ability to preserve layout and extract tables was beneficial for maintaining context in document chunks.
SimpleDirectoryReader (from llama_index.core): Facilitates reading and chunking documents stored in a local directory, enabling batch ingestion and preprocessing.

### Language Models and Embeddings
Ollama (ollama==0.4.4): Provides a streamlined way to run large language models (LLMs) like LLaMA 3 locally, using CPU or GPU. This allows inference and generation to be completely offline.
llama-index-llms-ollama (llama-index-llms-ollama==0.5.4): Integrates Ollama with the LlamaIndex framework, enabling seamless use of local LLMs in the RAG pipeline.
HuggingFaceEmbedding (llama-index-embeddings-huggingface==0.5.3): Utilized for generating high-quality vector embeddings from document chunks using transformer models, all within a local environment.

### Vector Database
ChromaDB (chromadb>=0.4.22): A local, fast, and easy-to-use vector database used to store and search document embeddings. This serves as the retrieval engine that powers the RAG pipeline.
llama-index-vector-stores-chroma (llama-index-vector-stores-chroma==0.4.1): A LlamaIndex-compatible wrapper around ChromaDB, used for managing document storage and similarity-based retrieval.

### Orchestration and RAG Pipeline
LlamaIndex (llama-index==0.12.33): The core framework used to coordinate document ingestion, embedding, indexing, retrieval, and query response generation. It simplifies the setup of a RAG pipeline by offering modular components for each stage.

### Auxiliary Tools
Python Typing (typing): Used for type hinting and code clarity.

Logging and Warning Management (logging, warnings): Ensures consistent logging behavior and suppression of non-critical warnings during execution.

OS Utilities (os): Handles file operations and environment setup.

Together, these components formed a cohesive and modular system for running a fully local, privacy-preserving document assistant. The project dependencies are managed via a requirements.txt file to ensure reproducibility and ease of setup.

## System Architecture

The system is designed as a local-first, document-aware assistant capable of ingesting PDF files, building a semantic index using vector embeddings, and responding to user questions through an interactive web interface. The architecture integrates several modular components—each responsible for ingestion, indexing, storage, retrieval, or interaction—into a unified Retrieval-Augmented Generation (RAG) pipeline. The key components and workflow are detailed below.

### Overall Flow
The application comprises five core stages:

1. Document Upload via the Streamlit front end.
2. Text Extraction using pdfplumber.
3. Document Indexing through LlamaIndex with HuggingFace embeddings.
4. Vector Storage and Retrieval using ChromaDB.
5. Question Answering using a locally hosted LLM via Ollama.

A diagram showing this end-to-end flow would include arrows from:

UI → Preprocessing → Embedding → Vector DB → LLM → UI

### Document Upload and Preprocessing
The Streamlit application provides a simple UI that allows users to upload multiple PDF files. These files are saved to a temporary directory (./docs/), and each page is rendered as an image using pdfplumber for optional display alongside the document interaction interface.

Each uploaded file is tracked using st.session_state to ensure that files are not reprocessed unnecessarily. Pages are extracted as images and stored for preview purposes.

### Embedding and Vector Storage
Once new files are uploaded, the documents are loaded and chunked using SimpleDirectoryReader, which scans the specified directory and prepares the documents for indexing. The semantic embeddings are generated using HuggingFaceEmbedding with the "BAAI/bge-base-en-v1.5" model—a sentence transformer well-suited for retrieval tasks.
These embeddings are then stored in a local vector database using ChromaDB, managed via ChromaVectorStore. This database supports approximate nearest neighbor search to efficiently retrieve document segments relevant to user queries.
All vector-related operations are orchestrated using the StorageContext from LlamaIndex, ensuring a clean separation between embedding logic and storage backend.

### Index and Query Engine Construction
Once the documents are embedded and stored, an index is created using VectorStoreIndex.from_documents. This index is coupled with an Ollama-powered LLM (llama3.2) to enable the query engine.

The query engine is configured to:
* Use top-3 most similar chunks based on vector similarity.
* Perform non-streaming inference locally.
* Handle LLM timeouts robustly (request_timeout=120.0).

The query engine is stored in session state and used for subsequent prompt-based interactions.

### User Interaction and Response Generation
The chatbot interface is embedded into the second column of the UI. It keeps a history of interactions using Streamlit’s session state and allows the user to input natural language questions. Once a prompt is submitted:
* The query is passed to the query engine.
* The top matching document segments are retrieved from Chroma.
* A response is generated using the Ollama-hosted LLM based on the retrieved context.

Responses are appended to the conversation history and rendered with role-based avatars (user and assistant).

### Caching and Optimization
To improve performance:
* PDF page images are cached using @st.cache_data.
* Warnings from third-party libraries (e.g., PyTorch) are suppressed.
* The protocol buffer implementation is explicitly set to python for compatibility.

Logging is also configured to aid in debugging and monitoring during development.

## Implementation Details
This section outlines the key implementation choices and techniques used to bring the system architecture to life, focusing on document processing, indexing, embedding, querying, and the user interface. The goal was to develop a modular, local-first application that is both easy to use and extensible.

### Project Initialization and Environment Setup
The project is implemented in Python and requires a small set of environment configurations:

Warnings related to torch.classes are suppressed to reduce console noise during execution.

The protocol buffer implementation is explicitly set to "python" via an environment variable to avoid compatibility issues.

Logging is configured using Python’s logging module for consistent runtime messaging.

```
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
```


### Streamlit-Based User Interface
The UI is built using Streamlit and configured with a minimal, wide layout:

```
st.set_page_config(
    page_title="Document Processing Agent",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
```

The interface is split into two main columns:
* Left Column (col1): Displays PDF previews and file upload controls.
* Right Column (col2): Houses the chatbot interface for querying indexed documents.

All persistent application states (e.g., uploaded files, query engine, message history) are maintained using st.session_state to preserve interactivity across multiple user actions.

### PDF Upload and Rendering
PDF files are uploaded using st.file_uploader, with support for multiple files. Each file is saved locally to the ./docs directory and processed if it hasn't already been indexed. A helper function uses pdfplumber to extract and cache each page as an image:

```
@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    with pdfplumber.open(file_upload) as pdf:
        return [page.to_image().original for page in pdf.pages]
```

### Embedding and Indexing
When new documents are detected, the system loads them using SimpleDirectoryReader and creates a VectorStoreIndex with:
* Embedding Model: BAAI/bge-base-en-v1.5, loaded via HuggingFaceEmbedding.
* Vector Store: ChromaDB, instantiated as an ephemeral client.
* Index Creation: VectorStoreIndex.from_documents() builds a searchable semantic index.

```
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader(custom_temp_dir, recursive=True).load_data()
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True,
)
```

### Query Engine and Local LLM
Once the index is built, a query engine is instantiated using a locally running LLM via the Ollama interface:

```
llm = Ollama(model="llama3.2", request_timeout=120.0)
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    streaming=False,
)
```

This engine performs similarity-based retrieval of document chunks and generates responses using the LLM. A user-defined function process_question() handles interaction with the engine and returns the generated text.

### Chat Interaction and State Management
The chatbot interface maintains conversational context using st.session_state["messages"]. Each user prompt and assistant response is rendered with role-based avatars and added to the persistent session:

```
if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    ...
    response = process_question(st.session_state["query_engine"], prompt)
    st.session_state["messages"].append({"role": "assistant", "content": response})
```

The interface checks if any documents are uploaded before answering queries and warns the user if not.

### Optimizations and Usability Features
Zoom Control: A slider allows dynamic resizing of rendered PDF page images.
Session Optimization: Uploaded files are tracked with a files_processed set to avoid reindexing.
Error Handling: Indexing exceptions are caught and reported via st.error.


## Results and Evaluations

TODO

## Conclusion
This project demonstrates the feasibility and practicality of building a completely local Retrieval-Augmented Generation (RAG) system for document-based question answering. By integrating open-source tools such as Streamlit, Ollama, LlamaIndex, ChromaDB, and Hugging Face embeddings, we developed a fully offline assistant that allows users to upload PDF documents and interact with them via natural language queries. The system preserves user privacy, avoids dependency on cloud services, and offers fine-grained control over how documents are parsed and processed.

From a technical standpoint, this work provided hands-on experience with vector indexing, local LLM hosting, and RAG architecture—all of which are key concepts in the evolving landscape of applied AI. It also highlights the importance of modular, transparent, and reproducible software design in building AI-driven applications that are both accessible and secure.

As large language models and vector search frameworks continue to mature, we see significant opportunities for expanding this work to support additional file formats, enhance retrieval performance, and build more conversational and context-aware assistants. This system provides a strong foundation for further experimentation in local AI and privacy-centric NLP applications.

##  References

1. [Ollama Documentation](https://ollama.com)
2. [Streamlit Documentation](https://docs.streamlit.io)
3. [LlamaIndex](https://www.llamaindex.ai)
4. [ChromaDB Documentation](https://docs.trychroma.com)
5. [Hugging Face Transformers](https://huggingface.co/BAAI/bge-base-en-v1.5)
6. [PDFPlumber](https://github.com/jsvine/pdfplumber)
7. [Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.](https://arxiv.org/abs/2005.11401)
8. [OpenAI. Introduction to Retrieval-Augmented Generation.](https://openai.com/blog/chatgpt-retrieval-plugin)
