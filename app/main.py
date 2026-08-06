import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.backends.ollama_backend import OllamaBackend
from app.ingestion import ingest_file, get_chunks


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edgeclaw")

app = FastAPI(title="EdgeClaw Lite", version="0.2.0")  # -> Creating a web server named app

# Create ONE backend instance when the server starts, and reuse it
# for every request (cheaper than building a new one each time).

backend = OllamaBackend(model="llama3.2:3b")  # -> creating an obj of class OllamaBackend

class ChatRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    path: str

class ChatResponse(BaseModel):
    answer: str
    query_length: int
    backend: str
    model: str
    latency_ms: float

@app.get("/health")  # -> decoreator : means when someone calls GET /health, run below func
def health():
    # Calling is_available() func of obj backend to check if Ollama is available
    return {"status": "ok", "backend_available": backend.is_available()}  

# Create POST /chat endpoint and tell FastAPI which response format to use.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    logger.info("Chat called with query: %s", request.query)

    # 1. Friendly error if Ollama isn't running.
    if not backend.is_available():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not reachable. Is the Ollama service running?",
        )
    
    # 2. Ask the real model (no retrieval/routing yet — that's later).
    result = backend.generate(request.query)

    # 3. If the model call itself failed, report it clearly.
    if result.error:
        raise HTTPException(status_code=502, detail=f"Model failed: {result.error}")
    
    # 4. Return the real answer plus useful info.
    return ChatResponse(
        answer=result.text,
        query_length=len(request.query),
        backend=result.backend,
        model=result.model,
        latency_ms=result.latency_ms,
    )

# When someone sends a POST request to /documents/ingest, run the function below
@app.post("/documents/ingest")
def documents_ingest(request: IngestRequest):
    logger.info("Ingest called for path: %s", request.path)
    try:
        summary = ingest_file(request.path)
        return {"status": "ok", **summary}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/documents/chunks")
def documents_chunks():
    chunks = get_chunks()
    # Return a short preview so the response isn't huge.
    preview = [
        {
            "doc_id": c["doc_id"],
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "text_preview": c["text"][:120],
        }
        for c in chunks[:20]
    ]
    return {"total_chunks": len(chunks), "preview": preview}