import os
import pickle
from pathlib import Path
from typing import List, Optional

# Ensure local calls (Ollama) bypass any proxy settings.
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import faiss
import ollama
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Resolve paths relative to this file so uvicorn can be run from anywhere
BASE_DIR = Path(__file__).parent

app = FastAPI(title="RAG Chat API")

def _cors_origins() -> List[str]:
    # Comma-separated list, e.g. "http://localhost:5173,http://127.0.0.1:5173"
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


# Setup CORS to allow the frontend to talk to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models and data
model: Optional[SentenceTransformer] = None
index = None
chunks: Optional[list[str]] = None
embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
ollama_model_name = os.getenv("OLLAMA_MODEL", "llama3")
ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
ollama_timeout_s = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))
top_k = int(os.getenv("TOP_K", "6"))

ollama_client = ollama.Client(host=ollama_host, timeout=ollama_timeout_s)

@app.on_event("startup")
def load_models():
    global model, index, chunks
    print("Loading models and FAISS index...")
    # Avoid network calls at runtime; expect the model to be cached locally.
    model = SentenceTransformer(embedding_model_name, local_files_only=True)

    index_path = BASE_DIR / "vector.index"
    chunks_path = BASE_DIR / "chunks.pkl"
    if not index_path.exists() or not chunks_path.exists():
        # Don't crash the server; expose a clear error on /health and /chat.
        index = None
        chunks = None
        print("WARNING: Missing index files. Run `python backend/build_index.py`.")
        return

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print("RAG models loaded successfully!")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []


@app.get("/health")
def health():
    return {
        "ok": model is not None and index is not None and chunks is not None,
        "embedding_model": embedding_model_name,
        "ollama_model": ollama_model_name,
        "top_k": top_k,
        "has_index": index is not None,
        "has_chunks": chunks is not None,
    }

@app.post("/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded yet.")
    if index is None or chunks is None:
        raise HTTPException(
            status_code=503,
            detail="Index not built. Run `python backend/build_index.py` to create vector.index and chunks.pkl.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    # Convert query to embedding
    query_embedding = model.encode([query]).astype("float32")

    # Search top-k similar chunks
    k = max(1, top_k)
    _, indices = index.search(query_embedding, k)

    # Retrieve context
    picked: List[str] = []
    for i in indices[0].tolist():
        if i is None or i < 0:
            continue
        if i >= len(chunks):
            continue
        picked.append(chunks[i])
    if not picked:
        return ChatResponse(answer="I couldn't find anything relevant in the indexed documents.", sources=[])
    # Keep the prompt reasonably small for local models.
    context = "\n\n".join(picked)[:8000]

    # Query Ollama
    response = ollama_client.chat(
        model=ollama_model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a RAG assistant. Use ONLY the provided context. "
                    "If the answer is not in the context, say you don't know. "
                    "Provide a detailed and accurate explanation."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
        ],
        options={
            # Prevent extremely long generations (helps UI responsiveness).
            "num_predict": 2048,
            "temperature": 0.2,
        },
    )

    answer = response.get("message", {}).get("content", "Sorry, I couldn't generate an answer.")
    # Return short source snippets to avoid huge payloads.
    sources = [s[:400] for s in picked]
    return ChatResponse(answer=answer, sources=sources)
