import os
os.environ["USER_AGENT"] = "rag-project/1.0"

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

# --- Load ---
all_docs = []
pdf_loader = DirectoryLoader("./my_docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
txt_loader = DirectoryLoader("./my_docs", glob="**/*.txt", loader_cls=TextLoader)
all_docs += pdf_loader.load()
all_docs += txt_loader.load()


# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
print(f"Created {len(chunks)} chunks")

# Embed and store
print("Embedding with FastEmbed...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"Done — {vectorstore._collection.count()} chunks stored")

# --- Quick test: search the vectorstore ---
print("\n--- Test retrieval ---")
results = vectorstore.similarity_search("what is constructor chaining", k=2)
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content[:300])
    print("Source:", doc.metadata.get("source"))