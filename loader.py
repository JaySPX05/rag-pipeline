from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
)

all_docs = []

# PDF
all_docs += PyPDFLoader(r"D:\rag-project\OOPS Assignment-1-2026.pdf").load()

# Text file
all_docs += TextLoader(r"D:\rag-project\test.txt").load()

# Website
all_docs += WebBaseLoader(r"https://en.wikipedia.org/wiki/Retrieval-augmented_generation").load()

# Folder
all_docs += DirectoryLoader(r"D:\rag-project\my_docs", glob="**/*.pdf", loader_cls=PyPDFLoader).load()

print(f"Total documents loaded: {len(all_docs)}")
for doc in all_docs[:3]:
    print("---")
    print(doc.metadata)
    print(doc.page_content[:200])