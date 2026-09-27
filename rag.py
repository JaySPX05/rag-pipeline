import os
os.environ["USER_AGENT"] = "rag-project/1.0"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# --- Load the existing vectorstore from disk ---
print("Loading vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
print(f"Vectorstore loaded — {vectorstore._collection.count()} chunks ready\n")

# --- Connect to Ollama ---
# --- Connect to Ollama ---
print("Connecting to Ollama...")
try:
    llm = OllamaLLM(model="qwen2:0.5b")
    print("Ollama connected!\n")
except Exception as e:
    print(f"Ollama connection failed: {e}")
    exit()

# --- Prompt template ---
prompt_template = PromptTemplate(
    template="""
You are a helpful assistant. You must answer ONLY from the context below.
Do not use any outside knowledge. Do not make anything up.
Keep your answer short and directly based on what the context says.
If the context does not contain the answer, say exactly: "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer (based only on the context above):""",
    input_variables=["context", "question"]
)

# --- Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def ask(question):
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        final_prompt = prompt_template.format(context=context, question=question)
        answer = llm.invoke(final_prompt)
        print(f"\nAnswer: {answer}")
        print("\nSources used:")
        for doc in docs:
            source = doc.metadata.get("source", "unknown").split("\\")[-1]
            page = doc.metadata.get("page", "")
            page_str = f" (page {page})" if page != "" else ""
            print(f"  - {source}{page_str}")
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"Error during ask: {e}")

# --- Interactive Q&A loop ---
print("RAG pipeline ready! Type your question below.")
print("Type 'quit' to exit\n")

try:
    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            print("Goodbye!")
            break
        if not question:
            continue
        print("\nThinking...\n")
        ask(question)
except Exception as e:
    print(f"Loop error: {e}")