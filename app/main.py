import logging
import datetime
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.backends.ollama_backend import OllamaBackend
from app.ingestion import ingest_file, get_chunks
from app import retriever, prompt_builder, vector_store, metrics, router, backend_registry, dispatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edgeclaw")

app = FastAPI(title="EdgeClaw Lite", version="0.2.0")  # -> Creating a web server named app

default_backend = OllamaBackend(model="llama3.2:3b")  # -> creating an obj of class OllamaBackend

class RouterConfig(BaseModel):
    mode : Literal["rule" , "similarity"]
    alpha : float = Field(ge=0.0, le=1.0)


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
    sources : List[dict]
    num_context_chunks : int
    route: str
    route_reason: str
    p_strong_wins : float | None = None
    alpha : float | None = None
    fallback_used : bool | None = None
    fallback_reason : str | None = None

@app.get("/health")  # -> decorator : means when someone calls GET /health, run below func
def health():
    # Calling is_available() func of obj backend to check if Ollama is available
    return {"status": "ok", "backend_available": backend_registry.get_backend("LOCAL_FALLBACK").is_available()}  

# get the current Router Config
@app.get("/router/config", response_model=RouterConfig)
def get_router_config() -> dict:
    return router.get_config()

# Use the PUT for same instance, but to update it. GET gives totally new instance
@app.put("/router/config", response_model = RouterConfig)
def update_router_config(config: RouterConfig) -> dict :
    updated_config = router.set_config(config.mode , config.alpha)
    return updated_config

# Create POST /chat endpoint and tell FastAPI which response format to use.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    logger.info("Chat called with query: %s", request.query)

    # 1. retrieve context -> get the route → build a grounded prompt → 
    # ask the model → return answer + sources
    chunks = retriever.retrieve(request.query, top_k= 3)

    # 2. Getting the query route and reason for the route
    context_size = 0
    for c  in chunks : 
        context_size = context_size + len(c["text"]) 
    route_dict = router.route(request.query, context_size)

    # 3. creating a prompt using RAG
    prompt = prompt_builder.build_rag_prompt(request.query , chunks)

    # 4. Routing the prompt appropriately via fallback models
    try : 
        dispatch_info = dispatcher.dispatch(route_dict["route"], prompt)
    except RuntimeError as e :
        raise HTTPException(status_code= 502, detail= f"All backends failed : {e}")
    
    result = dispatch_info["result"]

    source = []
    for chunk in chunks :
        source.append({
            "source" : chunk["source"],
            "chunk_id" : chunk["chunk_id"],
            "distance" : chunk["distance"]
        })

    # 5. Logging in the event metrics
    event_metrics = {
        "timestamp" : datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query" : request.query,
        "query_length" : route_dict["query_length"],
        "backend" : result.backend,
        "model" : result.model,
        "latency_ms" : result.latency_ms,
        "num_context_chunks" : len(chunks),
        "route" : route_dict["route"],
        "route_reason" : route_dict["reason"],
        "p_strong_wins" : route_dict.get("p_strong_wins"),
        "alpha" : route_dict.get("alpha"),
        "fallback_used" : dispatch_info["fallback_used"],
        "fallback_reason" : dispatch_info["fallback_reason"]        
    }

    metrics.log_chat_event(event_metrics)
    
    # 6. Return the real answer plus useful info.
    return ChatResponse(
        answer = result.text,
        query_length = route_dict["query_length"],
        backend = result.backend,
        model = result.model,
        latency_ms = result.latency_ms,
        sources = source,
        num_context_chunks = len(chunks),
        route = route_dict["route"],
        route_reason = route_dict["reason"],
        p_strong_wins = route_dict.get("p_strong_wins"),
        alpha = route_dict.get("alpha"),
        fallback_used = dispatch_info["fallback_used"],
        fallback_reason = dispatch_info["fallback_reason"] 
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

# When someone sends a POST request to /documents/reset, run the function below
@app.post("/documents/reset")
def documents_reset() -> dict :
    try:
        reset_res = vector_store.reset_collection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Collection Reset failed : {e}") 

    return {"status" : "ok", **reset_res}

@app.get("/metrics/recent")
def metrics_recent(limit:int = 20) -> dict :
    events = metrics.read_recent(limit)

    return {"count" : len(events), "events" : events}