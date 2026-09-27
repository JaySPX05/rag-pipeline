import os
os.environ["USER_AGENT"] = "rag-project/1.0"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load vectorstore ---
print("Loading vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f"Vectorstore loaded — {vectorstore._collection.count()} chunks ready")

# --- LLM ---
llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

# --- In-memory session store ---
sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""

@app.post("/chat")
async def chat(req: ChatRequest):
    # Get or create session
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Retrieve relevant chunks based on current message
    docs = retriever.invoke(req.message)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build messages list with system prompt + history + new message
    messages = [
        SystemMessage(content=f"""You are a smart AI assistant. Answer using the context below AND the conversation history.
If the answer is not in the context or history, say "I don't have enough information to answer that."
Keep your answers clear and concise.

Context from documents:
{context}""")
    ]

    # Add conversation history
    messages.extend(history)

    # Add current user message
    messages.append(HumanMessage(content=req.message))

    def stream():
        full_response = ""
        for chunk in llm.stream(messages):
            content = chunk.content
            full_response += content
            yield content

        # Save to history after streaming
        history.append(HumanMessage(content=req.message))
        history.append(AIMessage(content=full_response))

        # Keep last 10 exchanges (20 messages) to avoid token limits
        if len(history) > 20:
            sessions[session_id] = history[-20:]

        # Send sources
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
            page = doc.metadata.get("page", "")
            label = f"{source} (page {page})" if page != "" else source
            if label not in sources:
                sources.append(label)

        sources_str = "\n".join(sources)
        yield f"\n\n__SOURCES__{sources_str}__END__"

    return StreamingResponse(
        stream(),
        media_type="text/plain",
        headers={"X-Session-ID": session_id}
    )

@app.get("/")
async def root():
    return FileResponse("index.html")