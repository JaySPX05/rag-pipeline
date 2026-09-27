import os
os.environ["USER_AGENT"] = "rag-project/1.0"

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- Load ---
all_docs = []
all_docs += PyPDFLoader(r"D:\rag-project\OOPS Assignment-1-2026.pdf").load()
all_docs += TextLoader(r"D:\rag-project\test.txt").load()

# --- Chunk ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
print(f"Chunks ready to embed: {len(chunks)}")

# --- Embed and store ---
print("Loading embedding model... (first run downloads it, may take a minute)")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding chunks and saving to ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # saves to disk in your project folder
)

print(f"Done! {vectorstore._collection.count()} chunks stored in ChromaDB")

# --- Quick test: search the vectorstore ---
print("\n--- Test retrieval ---")
results = vectorstore.similarity_search("what is constructor chaining", k=2)
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content[:300])
    print("Source:", doc.metadata.get("source"))