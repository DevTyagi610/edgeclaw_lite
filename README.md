# EdgeClaw Lite
A local-first Retrieval-Augmented Generation (RAG) service built with Python and FastAPI.
It answers questions using local LLMs via Ollama, and is being extended with query
routing, multi-backend fallback, metrics logging, and benchmarking.
## Status
- [x] Step 1 — Environment + local model (Ollama, llama3.2:3b)
- [x] Step 2 — FastAPI skeleton (/health, /chat)
- [x] Step 3 — Real model answers via a clean backend interface
- [x] Step 4 — Document ingestion + chunking
- [ ] Step 5 — Embeddings + vector search (retrieval)
- [ ] Routing, fallback, metrics, benchmarking
## Tech stack
Python, FastAPI, Uvicorn, Ollama (llama3.2:3b)
## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs
API
- GET /health — service + backend status
- POST /chat — ask a question, get a model answer
- POST /documents/ingest — load a .txt/.md file and split into chunks
- GET /documents/chunks — preview stored chunks