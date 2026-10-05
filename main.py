import os, shutil, uuid
os.environ["USER_AGENT"] = "rag-project/1.0"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Load vectorstore ---
print("Loading vectorstore...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f"Vectorstore loaded — {vectorstore._collection.count()} chunks ready")

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, max_tokens=500)
sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""

# --- Stats endpoint ---
@app.get("/stats")
async def stats():
    count = vectorstore._collection.count()
    # Get unique source filenames
    try:
        results = vectorstore._collection.get(include=["metadatas"])
        sources = {}
        for meta in results["metadatas"]:
            src = meta.get("source","").split("\\")[-1].split("/")[-1]
            if src:
                sources[src] = sources.get(src, 0) + 1
        source_list = [{"name": k, "chunks": v} for k,v in sources.items()]
    except:
        source_list = []
    return {"chunks": count, "sources": source_list}

@app.delete("/document/{filename}")
async def delete_document(filename: str):
    try:
        # Get all IDs matching this source file
        results = vectorstore._collection.get(include=["metadatas"])
        ids_to_delete = []
        for i, meta in enumerate(results["metadatas"]):
            src = meta.get("source", "").split("\\")[-1].split("/")[-1]
            if src == filename:
                ids_to_delete.append(results["ids"][i])

        if not ids_to_delete:
            return {"success": False, "error": "Document not found"}

        vectorstore._collection.delete(ids=ids_to_delete)
        return {"success": True, "deleted": len(ids_to_delete), "filename": filename}

    except Exception as e:
        return {"success": False, "error": str(e)}

# --- Upload endpoint ---
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        tmp_dir = "./tmp_uploads"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, file.filename)

        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load based on file type
        ext = file.filename.lower().split(".")[-1]
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext in ["txt", "md"]:
            loader = TextLoader(tmp_path)
        else:
            return {"success": False, "error": "Unsupported file type"}

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Add to vectorstore
        vectorstore.add_documents(chunks)
        os.remove(tmp_path)

        return {"success": True, "chunks": len(chunks), "filename": file.filename}

    except Exception as e:
        return {"success": False, "error": str(e)}

# --- Chat endpoint ---
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    history = sessions[session_id]

    docs = retriever.invoke(req.message)
    context = "\n\n".join([doc.page_content[:300] for doc in docs])

    messages = [SystemMessage(content=f"""You are a smart AI assistant. Answer using the context below AND conversation history.
If the answer is not in the context or history, say "I don't have enough information to answer that."
Keep answers clear and concise.

Context:
{context}""")]

    messages.extend(history)
    messages.append(HumanMessage(content=req.message))

    def stream():
        full = ""
        for chunk in llm.stream(messages):
            content = chunk.content
            full += content
            yield content

        history.append(HumanMessage(content=req.message))
        history.append(AIMessage(content=full))
        if len(history) > 8:
            sessions[session_id] = history[-8:]

        sources = []
        for doc in docs:
            source = doc.metadata.get("source","unknown").split("\\")[-1].split("/")[-1]
            page = doc.metadata.get("page","")
            label = f"{source} (page {page})" if page != "" else source
            if label not in sources:
                sources.append(label)

        yield f"\n\n__SOURCES__{chr(10).join(sources)}__END__"

    return StreamingResponse(stream(), media_type="text/plain",
                             headers={"X-Session-ID": session_id})

@app.get("/")
async def root():
    return FileResponse("index.html")