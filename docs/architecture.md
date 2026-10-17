# EdgeClaw Lite Architecture

This document explains the current EdgeClaw Lite system flow in beginner-friendly terms.

## What EdgeClaw Lite does right now

EdgeClaw Lite is a local-first Retrieval-Augmented Generation (RAG) API service with
pluggable routing, multi-backend dispatch, fallback, observability, and evaluation.

Each `/chat` request:

1. Retrieves relevant chunks from an ingested document corpus.
2. Decides which backend tier should answer the query (a query **router**).
3. Dispatches the prompt to the chosen backend, with automatic fallback to secondary
   backends if the primary fails.
4. Logs the full request/response envelope to an append-only JSONL file.

The router has two modes, selectable at runtime via `GET/PUT /router/config`:

- **Rule mode** — simple length + keyword heuristics.
- **Similarity mode** — a RouteLLM-style Similarity-Weighted learned router that
  reuses the SentenceTransformer embeddings already used for retrieval. No training
  needed.

A manual evaluation harness scores stored benchmark runs against a hand-curated eval
set with per-query `must_contain` / `must_not_contain` assertions.

Current stack:

- API server: FastAPI
- Local LLM backends: Ollama with `llama3.2:1b` and `llama3.2:3b`
- Embedding model: SentenceTransformer `all-MiniLM-L6-v2`
- Vector store: ChromaDB persisted at `data/vector_store`
- Metrics log: `data/metrics/chat.jsonl` (JSON Lines, append-only)
- Chunk snapshots: `data/chunks/<doc_id>.json`
- Raw sample document: `data/raw/sample.md`

## Runtime flow

### 1. Server startup

When the server starts with uvicorn:

1. FastAPI creates the web app.
2. The backend registry is imported (no backends are instantiated yet — they are
   created lazily on first use).
3. The embedding model is not loaded immediately. It loads only when the first
   ingest, chat, or similarity-router call needs embeddings.
4. The Chroma vector store is also opened only when it is first needed.
5. The similarity router's preference set is loaded + embedded on the first `/chat`
   call in similarity mode.

Several objects follow the **lazy singleton** pattern — created once on first demand,
then reused:

- `app.embeddings.get_model`
- `app.vector_store.get_collection`
- `app.backend_registry.get_backend`
- `app.routers.similarity_router._get_preferences_cache`

### 2. Health check flow

Endpoint:

```text
GET /health
```

Purpose:

Check whether the API is alive and whether the default local backend is reachable.

Flow:

```text
GET /health
  -> health()
  -> backend_registry.get_backend("LOCAL_FALLBACK").is_available()
  -> Ollama client list request
  -> return {status, backend_available}
```

Expected successful response:

```json
{
  "status": "ok",
  "backend_available": true
}
```

### 3. Document ingest flow

Endpoint:

```text
POST /documents/ingest
```

Purpose:

Read a local document, split it into chunks, embed those chunks, and store them in
the vector database.

Example request:

```json
{"path": "data/raw/sample.md"}
```

High-level flow:

```text
file path
  -> read file text
  -> split text into chunks (chunk_size=800, overlap=100)
  -> embed every chunk
  -> store chunk vectors in ChromaDB
  -> save chunk JSON copy on disk
  -> return ingest summary
```

Detailed call flow:

```text
main.documents_ingest(request)
  -> ingestion.ingest_file(path)
       -> ingestion.load_document(path)
       -> compute STABLE doc_id = md5(abspath(path))[:8]
       -> vector_store.delete_by_doc_id(doc_id)   # idempotency
       -> chunking.chunk_text(text)
       -> build chunk records with doc_id, source, chunk_id, text
       -> save records to data/chunks/<doc_id>.json
       -> embeddings.embed_texts(chunk_texts)
            -> embeddings.get_model()
            -> SentenceTransformer.encode(chunks)
       -> vector_store.add_chunks(records, embedded_texts)
            -> vector_store.get_collection()
            -> Chroma collection.add(...)
       -> return summary
```

Expected successful response shape:

```json
{
  "status": "ok",
  "doc_id": "7f528729",
  "source": "sample.md",
  "num_chunks": 4,
  "saved_to": "data/chunks/7f528729.json",
  "num_embedded": 4
}
```

### 4. Chat / RAG flow

Endpoint:

```text
POST /chat
```

Purpose:

Answer a user question using retrieved document context, routed to the appropriate
backend tier with automatic fallback on failure.

Example request:

```json
{"query": "What wand did Harry finally get?"}
```

High-level flow:

```text
user question
  -> retrieve top-k chunks from ChromaDB
  -> compute route decision (rule or similarity router)
  -> pick concrete backend for that route from the registry
  -> build a grounded RAG prompt
  -> dispatch prompt to backend (with fallback chain on failure)
  -> log metrics event to JSONL
  -> return answer + sources + route metadata + fallback info
```

Detailed call flow:

```text
main.chat(request)
  -> retriever.retrieve(request.query, top_k=3)
       -> embeddings.embed_query(query)
            -> embeddings.get_model()
            -> SentenceTransformer.encode(query)
       -> vector_store.query(query_embedding, top_k)
            -> Chroma collection.query(...)
            -> return chunks with text, source, doc_id, chunk_id, distance
  -> compute context_size = sum(len(chunk.text) for chunk in chunks)
  -> router.route(query, context_size)
       -> if ROUTER_MODE == "similarity":
            -> similarity_router.route(query, alpha, context_size)
       -> else:
            -> rules.classify(query)
       -> return {route, reason, query_length, context_size,
                  p_strong_wins (sim only), alpha (sim only)}
  -> prompt_builder.build_rag_prompt(query, chunks)
       -> system instructions + retrieved context + question
  -> dispatcher.dispatch(route_label, prompt)
       -> for each backend in get_fallback_chain(route_label):
            -> try is_available() + generate()
            -> if either fails, record reason and try next
       -> return {result, final_route, chain_used,
                  fallback_used, fallback_reason}
  -> metrics.log_chat_event({...})
       -> append JSON line to data/metrics/chat.jsonl
  -> return ChatResponse
```

Expected successful response shape:

```json
{
  "answer": "...",
  "query_length": 32,
  "backend": "ollama",
  "model": "llama3.2:1b",
  "latency_ms": 1234.5,
  "sources": [
    {"source": "sample.md", "chunk_id": 2, "distance": 0.75}
  ],
  "num_context_chunks": 3,
  "route": "LOCAL_FAST",
  "route_reason": "Short query <=80 length, can be done by local Fast model",
  "p_strong_wins": null,
  "alpha": null,
  "fallback_used": false,
  "fallback_reason": null
}
```

### 5. Query routing + backend dispatch + fallback

This is the layer between retrieval and generation. It consists of three
cooperating modules:

#### 5a. Router — pick a label

`app/router.py::route(query, context_size)` returns a dict:

```text
{"route": <label>, "reason": <string>,
 "query_length": <int>, "context_size": <int>,
 "p_strong_wins": <float|None>, "alpha": <float|None>}
```

Where `<label>` is one of `"LOCAL_FAST"`, `"LOCAL_FALLBACK"`, `"CLOUD"`.

The router switches between two implementations based on the module-level
`ROUTER_MODE` constant (`"rule"` by default).

- **Rule router** (`app/rules.py::classify`): pure heuristics.
  - Contains any of `summarize`, `analyze`, `compare` → `CLOUD`
  - Else, query length ≤ 80 chars → `LOCAL_FAST`
  - Else → `LOCAL_FALLBACK`

- **Similarity router** (`app/routers/similarity_router.py`): RouteLLM-style
  Similarity-Weighted method (Ong et al., ICLR 2025, Section 4.2), simplified to
  skip the max-normalization step for a small local preference set.
  - At startup, load `benchmarks/preferences.jsonl` (hand-labeled
    `{query, label in {strong_wins, weak_wins, tie}}`) and embed each query
    with MiniLM. Cache in memory.
  - On query: embed the query, compute cosine similarity vs every preference,
    weight by `gamma^(1 + sim)` with `gamma=10`, take weighted mean of labels
    to produce `P(strong_wins | query)`.
  - Compare to threshold `alpha`:
    - `P >= alpha` → `CLOUD`
    - `P <  alpha` → `LOCAL_FAST`
    - Empty query / missing preferences → `LOCAL_FALLBACK` (safe fallback)

Mode and threshold are inspectable and mutable at runtime:

```text
GET  /router/config    -> {"mode": "rule", "alpha": 0.5}
PUT  /router/config    body: {"mode": "similarity", "alpha": 0.5}
```

#### 5b. Backend registry — map label to concrete backend

`app/backend_registry.py` owns the `label -> backend` mapping and lazy
instantiation:

```python
BACKEND_CONFIG = {
    "LOCAL_FAST":     {"type": "ollama", "model": "llama3.2:1b",
                       "fallbacks": ["LOCAL_FALLBACK"]},
    "LOCAL_FALLBACK": {"type": "ollama", "model": "llama3.2:3b",
                       "fallbacks": ["LOCAL_FAST"]},
    "CLOUD":          {"type": "ollama", "model": "llama3.2:3b",
                       "fallbacks": ["LOCAL_FALLBACK", "LOCAL_FAST"]},
}
```

- `get_backend(label)` — returns a cached backend instance for the label.
  Invalid labels are normalized to the default (`LOCAL_FALLBACK`) with a warning.
- `get_fallback_chain(label)` — returns `[primary] + fallbacks`, the full
  list the dispatcher will walk in order.

The design is deliberately hardware-agnostic. Route labels are *abstract tiers*;
the registry decides which concrete model / device / API serves each tier.
Adding a Snapdragon NPU backend or a cloud API would mean adding a new backend
type and one line to `BACKEND_CONFIG` — no router or dispatcher changes.

#### 5c. Dispatcher — try backends in order

`app/dispatcher.py::dispatch(route_label, prompt)` executes the fallback chain:

```text
for each label in get_fallback_chain(route_label):
    backend = get_backend(label)
    if not backend.is_available():
        record "<label> not available", continue
    result = backend.generate(prompt)
    if result.error:
        record "<label> error: ...", continue
    # success
    return {result, final_route=label, chain_used,
            fallback_used=(index>0), fallback_reason=joined_failures_or_None}

# whole chain failed
raise RuntimeError(summary of attempts)
```

The handler in `main.py::chat` catches `RuntimeError` and returns HTTP 502 with
the full attempt log. This replaces the pre-Phase-8 behavior where any single
backend failure caused a 503.

### 6. Observability — metrics + benchmark + eval

Three cooperating pieces, each with a single responsibility.

#### 6a. Metrics log

`app/metrics.py::log_chat_event(event)` appends one JSON line to
`data/metrics/chat.jsonl` for every successful `/chat` response. Each line
contains: timestamp, query, query_length, backend, model, latency_ms,
num_context_chunks, route, route_reason, p_strong_wins, alpha, fallback_used,
fallback_reason.

The write is wrapped in try/except so metrics failure can never break `/chat`.

`app/metrics.py::read_recent(limit)` reads the newest `limit` events for the
`/metrics/recent?limit=N` HTTP endpoint. Returns newest-first, slice-then-reverse
to avoid pulling the whole file into memory on every call.

#### 6b. Benchmark runner

`benchmarks/run_benchmark.py` is a standalone HTTP client. It:

1. Loads a fixed list of queries from `benchmarks/queries.yaml`.
2. Health-checks the server.
3. POSTs each query to `/chat`.
4. Writes two artifacts into `benchmarks/results/`:
   - `results_<UTC-TIMESTAMP>.csv` — one row per query with full response fields
     + keyword-match stats
   - `results_<UTC-TIMESTAMP>.json` — same data in machine-friendly form

Each row stores both `answer_preview` (first 120 chars, for readable CSVs) and
`answer_full` (complete response, for faithful evaluation).

Timestamps use `%Y-%m-%dT%H-%M-%SZ` so filenames are safe on all operating
systems. Nothing gets overwritten; every run leaves a dated artifact for
comparison across configurations.

#### 6c. Manual evaluation

`benchmarks/evaluate.py` scores a stored benchmark run against
`benchmarks/eval.yaml`:

- `eval.yaml` contains per-query ground-truth metadata:
  `{id, category, reference_answer, must_contain, must_not_contain}`.
- The script joins by `id`, scans `answer_full` for required/forbidden
  substrings (case-insensitive), and produces per-query verdicts.
- Output: global totals, per-category pass rates, and a per-row detail line
  showing which required terms were missing or forbidden terms hallucinated.

The script does not call `/chat` — it only reads captured results. This keeps
evaluation reproducible: the same results file can be re-scored any number of
times against evolving eval sets.

## Function responsibilities

### `app/main.py`

Owns the FastAPI endpoints.

Responsibilities:

- Define request and response models.
- Expose `/health`, `/chat`, `/documents/ingest`, `/documents/chunks`,
  `/documents/reset`, `/router/config` (GET+PUT), `/metrics/recent`.
- Orchestrate the full RAG + routing + dispatch flow for `/chat`.
- Convert internal results into API responses.
- Convert errors into HTTP status codes (502 when all backends fail, 503 only
  used by `/health` for the default backend).

### `app/backends/ollama_backend.py`

Owns communication with Ollama.

Responsibilities:

- Create an Ollama client.
- Check whether Ollama is reachable.
- Send prompts to the selected Ollama model.
- Return a standard backend response.

### `app/backends/base.py`

Defines the backend contract.

Responsibilities:

- Define the standard `BackendResponse` shape.
- Define `BaseBackend` with abstract `generate` + `is_available` methods.

### `app/ingestion.py`

Owns document ingestion.

Responsibilities:

- Validate file path and file type.
- Read `.txt` and `.md` files.
- Compute stable `doc_id` from the file path's md5 so re-ingesting the same
  file is idempotent.
- Call `vector_store.delete_by_doc_id` before adding, replacing any prior chunks.
- Chunk the text, embed each chunk, and store in ChromaDB.
- Save chunk JSON snapshots for inspection.

### `app/chunking.py`

Owns text splitting.

Responsibilities:

- Split long text into overlapping chunks (default 800 chars with 100 overlap).

### `app/embeddings.py`

Owns embedding model loading and embedding generation.

Responsibilities:

- Lazy-load the SentenceTransformer model once (`get_model`).
- Convert document chunks into vectors (`embed_texts`).
- Convert user queries into vectors (`embed_query`).

### `app/vector_store.py`

Owns ChromaDB access.

Responsibilities:

- Lazy-open or create the persistent ChromaDB collection.
- Add embedded document chunks.
- Query the vector store for similar chunks.
- `delete_by_doc_id(doc_id)` for idempotent ingestion.
- `reset_collection()` for a hard reset via `POST /documents/reset`.

### `app/retriever.py`

Owns retrieval orchestration.

Responsibilities:

- Convert the user query into an embedding.
- Ask the vector store for the top matching chunks.
- Return the retrieved chunks to the chat endpoint.

### `app/prompt_builder.py`

Owns prompt construction.

Responsibilities:

- Build a RAG prompt that tells the model to answer using only the retrieved
  context.
- Handle the case where no chunks were retrieved.

### `app/rules.py`

Owns rule-based routing.

Responsibilities:

- Pure function `classify(query) -> {route, reason}` with no imports from the
  rest of the app.
- Keyword matching (`summarize`/`analyze`/`compare`) and length thresholds
  decide the route label.

### `app/router.py`

Owns the router dispatch surface.

Responsibilities:

- Expose `route(query, context_size)` to the rest of the app.
- Pick between rule-based and similarity-based implementations based on
  `ROUTER_MODE`.
- `get_config()` / `set_config(mode, alpha)` for the HTTP config endpoints.
- Graceful fallback to rule mode on invalid `ROUTER_MODE`.

### `app/routers/similarity_router.py`

Owns the RouteLLM-style Similarity-Weighted router.

Responsibilities:

- Load + embed the hand-labeled preference set once (`_load_preferences`).
- Compute `P(strong_wins | query)` for a new query using cosine similarity and
  Bradley–Terry weighting.
- Return a routing decision dict compatible with the rule router's shape, plus
  `p_strong_wins` and `alpha` for observability.

### `app/backend_registry.py`

Owns the label → backend mapping and fallback chain definitions.

Responsibilities:

- `BACKEND_CONFIG` as single source of truth.
- `get_backend(label)` with lazy instantiation and typo tolerance.
- `get_fallback_chain(label)` → `[primary] + fallbacks`.

### `app/dispatcher.py`

Owns the retry-with-fallback loop.

Responsibilities:

- Walk the fallback chain for a route label.
- Catch is_available/generate failures per backend and continue to the next.
- Return `{result, final_route, chain_used, fallback_used, fallback_reason}`.
- Raise `RuntimeError` with full attempt log only when every backend failed.

### `app/metrics.py`

Owns per-request metrics logging + retrieval.

Responsibilities:

- `log_chat_event(event)` — append one JSON line to `data/metrics/chat.jsonl`.
  Fail-safe; never propagates exceptions.
- `read_recent(limit)` — return the N newest events, newest-first, for
  `/metrics/recent`.

### `benchmarks/run_benchmark.py`

Standalone HTTP client for the benchmark loop.

Responsibilities:

- Load `benchmarks/queries.yaml`.
- Health-check the server and POST each query to `/chat`.
- Compute keyword-match stats per query.
- Write timestamped CSV + JSON artifacts to `benchmarks/results/`.

### `benchmarks/evaluate.py`

Scores a stored benchmark run against an eval set.

Responsibilities:

- Load `benchmarks/eval.yaml` + a specific results JSON (or the latest).
- For each row, check `must_contain` and `must_not_contain` against
  `answer_full`.
- Print a human-readable report (global, per-category, per-row).
- Return a summary dict (`scored`, `passed`, `failed`, `skipped`, `pass_rate`,
  `by_category`, `per_row`); optionally write to `--out` JSON.

## Important current behavior

### ChromaDB persists across server restarts

The vector store lives on disk at:

```text
data/vector_store
```

Stored chunks remain even if you stop and restart uvicorn.

### `/documents/chunks` is only in-memory

The endpoint `/documents/chunks` reads from the `_INGESTED` list inside
`ingestion.py`. That list is in memory only. So after a server restart:

- `/documents/chunks` may look empty.
- `/chat` can still retrieve from ChromaDB.

This is expected.

### Ingestion is idempotent

Each file's `doc_id` is derived deterministically from its absolute path via
md5. On every ingest, `vector_store.delete_by_doc_id(doc_id)` runs *before* the
new chunks are added. Result: ingesting the same file N times produces exactly
one set of chunks in Chroma, not N copies.

The pre-ingestion in-memory list `_INGESTED` is also kept clean using
slice-assignment to filter out old entries with the same `doc_id`.

For a hard wipe of the whole collection, `POST /documents/reset` calls
`vector_store.reset_collection()`.

### Router mode is runtime-mutable

`ROUTER_MODE` and `ROUTER_ALPHA` live as module-level constants in
`app/router.py` but are inspected and mutated via `GET/PUT /router/config`.
Changes take effect immediately, no restart. The underlying similarity
preference embeddings are cached in memory, so the first similarity-mode
`/chat` after startup has a slight warm-up cost.

### `answer_preview` vs `answer_full` in benchmark records

Benchmark rows store both. `answer_preview` is the first 120 characters (for
readable CSV cells). `answer_full` is the complete model response (for
faithful eval scoring). The evaluator reads `answer_full` and falls back to
`answer_preview` for pre-Phase-11 result files.

## Current known test case

Sample document:

```text
data/raw/sample.md
```

Question:

```text
What wand did Harry finally get?
```

Expected grounded answer:

```text
Holly and phoenix feather, eleven inches.
```

If the answer is wrong or hedging, likely causes are:

1. Routed to a weak backend (`llama3.2:1b`) that hallucinates a rejected wand
   like `"ebony and unicorn hair"`.
2. Context chunk 2 (which contains the answer) was not retrieved.
3. The document was not ingested.

The eval set catches case 1 via `must_not_contain: [ebony, maple, Beechwood]`.

## Mental model

Think of the system as six concentric boxes:

```text
Documents -> Embeddings -> Vector Store -> Routing -> Dispatch -> LLM Answer
```

Or more fully:

```text
Ingest time:
Document -> chunks -> chunk embeddings -> ChromaDB

Question time:
Question -> query embedding -> retrieve chunks from ChromaDB
         -> router picks label (rule or SW)
         -> dispatcher picks concrete backend from label's chain
         -> prompt_builder assembles grounded prompt
         -> backend.generate -> answer
         -> metrics.log_chat_event
         -> response includes route + fallback metadata
```

Observability runs in parallel:

```text
Every /chat ---append---> data/metrics/chat.jsonl
                              -> GET /metrics/recent (live inspection)
                              -> benchmarks/run_benchmark (replay queries.yaml)
                              -> benchmarks/evaluate (score vs eval.yaml)
```

Each layer has one responsibility and can be swapped independently. Replace the
rule router with the SW router? Flip `ROUTER_MODE`. Add a Snapdragon NPU backend?
One entry in `BACKEND_CONFIG` + a new `BaseBackend` subclass. Add file upload?
A new endpoint in `main.py` that writes to `data/raw/` and calls `ingest_file`.
