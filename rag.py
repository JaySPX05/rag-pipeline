import os
os.environ["USER_AGENT"] = "rag-project/1.0"

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- Load vectorstore ---
print("Loading vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f"Vectorstore loaded — {vectorstore._collection.count()} chunks ready\n")

# --- LLM ---
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# --- Prompt ---
prompt_template = PromptTemplate(
    template="""You are a smart AI assistant. Answer ONLY from the context below.
If the answer is not in the context, say "I don't have enough information to answer that."
Keep your answer clear and concise.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

def ask(question):
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        final_prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke(final_prompt)
        print(f"\nAnswer: {response.content}")
        print("\nSources used:")
        for doc in docs:
            source = doc.metadata.get("source", "unknown").split("\\")[-1]
            page = doc.metadata.get("page", "")
            page_str = f" (page {page})" if page != "" else ""
            print(f"  - {source}{page_str}")
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"Error: {e}")

# --- Interactive loop ---
print("RAG pipeline ready! Type your question below.")
print("Type 'quit' to exit\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "quit":
        print("Goodbye!")
        break
    if not question:
        continue
    print("\nThinking...\n")
    ask(question)