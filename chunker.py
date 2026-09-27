import os
os.environ["USER_AGENT"] = "rag-project/1.0"

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Load documents (same as before) ---
all_docs = []
all_docs += PyPDFLoader(r"D:\rag-project\OOPS Assignment-1-2026.pdf").load()
all_docs += TextLoader("test.txt").load()

print(f"Documents before chunking: {len(all_docs)}")

# --- Chunk them ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # max characters per chunk
    chunk_overlap=50,     # overlap between chunks to preserve context
    separators=["\n\n", "\n", ".", " "]  # tries to split at natural boundaries
)

chunks = splitter.split_documents(all_docs)

print(f"Chunks after splitting: {len(chunks)}")
print(f"\n--- Sample chunk 1 ---")
print(chunks[0].page_content)
print(f"\n--- Sample chunk 2 ---")
print(chunks[1].page_content)
print(f"\n--- Metadata ---")
print(chunks[0].metadata)