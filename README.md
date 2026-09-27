# Local RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) pipeline built with Python.
Ask questions about your own documents — PDFs, text files, and websites —
using a local LLM with no data leaving your machine.

## Tech Stack
- **LangChain** — orchestration and document loading
- **ChromaDB** — local vector store
- **Sentence Transformers** — embedding model (all-MiniLM-L6-v2)
- **Ollama** — local LLM runner
- **Python 3.10+**

## Project Structure
rag-project/
├── loader.py # Load documents from PDFs, text files, URLs
├── chunker.py # Split documents into chunks
├── embedder.py # Embed chunks and store in ChromaDB
├── rag.py # Interactive Q&A loop
├── requirements.txt
└── README.md


## Setup

1. Clone the repo
```bash
git clone https://github.com/JaySPX05/rag-pipeline.git
cd rag-pipeline
```

2. Create and activate a virtual environment
```bash
python -m venv rag-env
source rag-env/bin/activate  # Windows: rag-env\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install and start Ollama from https://ollama.com then pull a model
```bash
ollama pull qwen2:0.5b   # lightweight, works on 8GB RAM
# or
ollama pull llama3       # better quality, needs 16GB RAM
```

5. Add your documents to the project folder

6. Run the pipeline
```bash
python loader.py     # test your loaders
python embedder.py   # embed and index your documents
python rag.py        # start asking questions
```

## Usage
RAG pipeline ready! Type your question below.
Type 'quit' to exit

You: What is the remote work policy?
Thinking...
Answer: Employees may work remotely up to 3 days per week with manager approval.
Sources used:

test.txt


## Features
- Load PDFs, text files, markdown, and websites
- Chunk documents with configurable size and overlap
- Store embeddings locally in ChromaDB (persists to disk)
- Semantic search retrieval with source citations
- Runs 100% locally — no data sent to any API

## Adding your own documents
Place your files in the project folder:
- PDFs → update `PyPDFLoader("your_file.pdf")` in `embedder.py`
- Text files → update `TextLoader("your_file.txt")`
- Websites → update the URL in `WebBaseLoader(...)`