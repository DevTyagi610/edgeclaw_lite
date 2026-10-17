# EdgeClaw Lite

A local-first Retrieval-Augmented Generation (RAG) service with pluggable query
routing, multi-backend dispatch, fallback logic, metrics logging, benchmarking,
and manual evaluation.

Built end-to-end to bridge theoretical ML/DL knowledge and practical AI
engineering: a measurable, resumeable project where every piece of plumbing
was built from scratch and every behavior is observable.

## Features

- **End-to-end RAG pipeline** — document ingestion → chunking → embeddings →
  Chroma vector store → retrieval → prompt building → local LLM generation.
- **Pluggable query routing**:
  - Rule-based (length + keyword heuristics).
  - RouteLLM-style Similarity-Weighted learned router (based on Ong et al.,
    *RouteLLM*, ICLR 2025) — zero training, uses the same MiniLM embeddings as
    retrieval.
  - Mode and threshold are switchable at runtime via HTTP.
- **Multi-backend dispatcher** — route labels map to concrete Ollama models
  (currently `llama3.2:1b` for `LOCAL_FAST`, `llama3.2:3b` for
  `LOCAL_FALLBACK` and `CLOUD`). Hardware-agnostic: adding a Snapdragon NPU
  backend or a cloud API is a config change, not a refactor.
- **Fallback logic** — if the chosen backend fails, the dispatcher walks a
  per-route fallback chain and silently recovers.
- **Observable** — every `/chat` request appends a structured JSON line to
  `data/metrics/chat.jsonl` and is retrievable via `GET /metrics/recent`.
- **Idempotent ingestion** — ingesting the same document N times produces one
  copy in the vector store (stable path-based `doc_id` + delete-before-add).
- **Benchmark + evaluation** — standalone CLI benchmarks (timestamped CSV +
  JSON outputs) and a manual evaluation harness that scores results against a
  hand-curated eval set with `must_contain` / `must_not_contain` assertions.

## Architecture at a glance

```text
POST /chat
  → retrieve top-k chunks from Chroma
  → router picks a route label         (rule mode OR similarity mode)
  → backend registry → concrete backend (LOCAL_FAST / LOCAL_FALLBACK / CLOUD)
  → prompt_builder assembles RAG prompt
  → dispatcher calls backend, with per-route fallback chain on failure
  → metrics.log_chat_event → data/metrics/chat.jsonl (append-only)
  → ChatResponse(answer, sources, route, fallback_used, p_strong_wins, alpha, ...)
```

Each layer has one responsibility and can be swapped independently. See
[`docs/architecture.md`](docs/architecture.md) for the full flow diagrams and
per-module responsibilities.

## Tech stack

- **API server**: FastAPI + Uvicorn (Python 3.10+)
- **Local LLMs**: Ollama (`llama3.2:1b`, `llama3.2:3b`)
- **Embeddings**: SentenceTransformer `all-MiniLM-L6-v2`
- **Vector store**: ChromaDB (persistent)
- **Benchmark/eval**: `requests`, `pyyaml`, `numpy`, `csv`, `json`

## Repository layout

```text
edgeclaw_lite/
├── app/
│   ├── main.py                    FastAPI endpoints + request orchestration
│   ├── backends/
│   │   ├── base.py                BaseBackend contract + BackendResponse
│   │   └── ollama_backend.py      Ollama HTTP adapter
│   ├── routers/
│   │   └── similarity_router.py   RouteLLM Similarity-Weighted router
│   ├── chunking.py                Overlapping chunker (800/100 by default)
│   ├── ingestion.py               Idempotent document ingestion
│   ├── embeddings.py              MiniLM lazy singleton
│   ├── vector_store.py            Chroma collection access + reset + delete
│   ├── retriever.py               Embed + top-k query
│   ├── prompt_builder.py          Grounded RAG prompt construction
│   ├── rules.py                   Rule-based classify()
│   ├── router.py                  Router mode selector + GET/PUT config
│   ├── backend_registry.py        Label → backend map + fallback chains
│   ├── dispatcher.py              Retry-with-fallback loop
│   └── metrics.py                 JSONL append + read_recent
├── benchmarks/
│   ├── queries.yaml               Test queries (id, keywords)
│   ├── preferences.jsonl          Hand-labeled SW router preferences
│   ├── eval.yaml                  Per-query must_contain / must_not_contain
│   ├── run_benchmark.py           CLI: POST /chat per query → timestamped CSV+JSON
│   └── evaluate.py                CLI: score latest results JSON vs eval.yaml
├── data/
│   ├── raw/sample.md              Example document (gitignored except sample)
│   ├── vector_store/              Chroma persistent store (gitignored)
│   ├── chunks/                    Per-doc chunk JSON snapshots (gitignored)
│   └── metrics/chat.jsonl         Append-only request log (gitignored)
├── docs/architecture.md           Full architecture doc
└── README.md
```

## API reference

| Method | Path                   | Purpose                                        |
|--------|------------------------|------------------------------------------------|
| GET    | `/health`              | Service + default backend availability         |
| POST   | `/chat`                | Ask a question → grounded answer + metadata    |
| POST   | `/documents/ingest`    | Load a `.txt` / `.md` file → chunks + vectors  |
| GET    | `/documents/chunks`    | Preview in-memory chunk list                   |
| POST   | `/documents/reset`     | Hard-reset Chroma collection                   |
| GET    | `/router/config`       | Current `{mode, alpha}`                        |
| PUT    | `/router/config`       | Change `{mode, alpha}` at runtime              |
| GET    | `/metrics/recent?limit=N` | Last N JSONL events, newest-first          |

Swagger UI at `http://127.0.0.1:8000/docs` once the server is running.

## Quick start

```bash
# 1. Clone and set up the virtual environment
git clone https://github.com/DevTyagi610/edgeclaw_lite.git
cd edgeclaw_lite
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Make sure Ollama is running and pull the models
ollama pull llama3.2:1b
ollama pull llama3.2:3b

# 3. Start the server
uvicorn app.main:app --reload

# 4. In a second terminal: ingest the sample doc and ask a question
curl -s -X POST http://127.0.0.1:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "data/raw/sample.md"}'

curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What wand did Harry finally get?"}' | python3 -m json.tool
```

## Sample `/chat` response

```json
{
  "answer": "Harry got a holly and phoenix feather wand, eleven inches, nice and supple.",
  "query_length": 32,
  "backend": "ollama",
  "model": "llama3.2:1b",
  "latency_ms": 2890.4,
  "sources": [
    {"source": "sample.md", "chunk_id": 2, "distance": 0.75},
    {"source": "sample.md", "chunk_id": 1, "distance": 0.76},
    {"source": "sample.md", "chunk_id": 3, "distance": 0.78}
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

## Switching to the RouteLLM-style similarity router

```bash
curl -s -X PUT http://127.0.0.1:8000/router/config \
  -H "Content-Type: application/json" \
  -d '{"mode": "similarity", "alpha": 0.5}'
```

Subsequent `/chat` responses include `p_strong_wins` (predicted probability
that a strong model would win on this query) and `alpha` (current threshold).

## Benchmark + evaluation

```bash
# 1. Run the fixed benchmark suite
python -m benchmarks.run_benchmark
# → writes benchmarks/results/results_<UTC-TIMESTAMP>.{csv,json}

# 2. Score the latest run against the hand-curated eval set
python -m benchmarks.evaluate
```

### Sample eval report

```text
EdgeClaw Lite — eval report
  eval file   : benchmarks/eval.yaml
  results file: benchmarks/results/results_2026-10-17T07-43-52Z.json

  scored  : 5
  passed  : 1
  failed  : 4
  skipped : 0
  pass rate: 20.0%

  by category:
    analyze      : 1/2 (50.0%)
    domain       : 0/1 (0.0%)
    summarize    : 0/1 (0.0%)
    trivia       : 0/1 (0.0%)

  details:
    [fail   ]  wand_final        (domain)      missing: ['holly', 'phoenix', 'eleven']  hallucinated: ['ebony']
    [fail   ]  wand_shop_owner   (trivia)      missing: ['Ollivander']
    [fail   ]  phoenix_scar      (analyze)     missing: ['phoenix', 'feather', 'brother']
    [fail   ]  wand_summary      (summarize)   missing: ['holly']
    [pass   ]  long_story        (analyze)
```

The `must_not_contain` assertion catches "confidently wrong" answers that
keyword-only matching would reward — e.g., the `wand_final` row above where
the small model asserted an `ebony` wand (actually rejected by Ollivander in
the source document). This is where the eval harness pays for itself.

## Development journey (12 phases)

| # | Phase                              | Delivered                                                 |
|---|------------------------------------|-----------------------------------------------------------|
| 0 | Environment + concept              | Py 3.10 venv, Ollama, local models                        |
| 1 | FastAPI skeleton                   | `/health`, mock `/chat`                                   |
| 2 | Ollama backend adapter             | `BaseBackend`, `OllamaBackend`, 503/502 guards            |
| 3 | Document ingestion                 | Chunker, `/documents/ingest`, JSON snapshots              |
| 4 | Embeddings + vector store          | MiniLM, Chroma persistent store                           |
| 5 | RAG chat pipeline                  | retriever + prompt_builder + wired `/chat`                |
| 6a | Rule-based query router           | `rules.py`, `router.py`                                   |
| 6b | RouteLLM Similarity-Weighted      | `routers/similarity_router.py` + hand-labeled prefs       |
| 7 | Multi-backend dispatcher           | `backend_registry.py` with lazy label→backend cache       |
| 8 | Fallback logic                     | `dispatcher.py` + fallback chains + per-attempt reasons   |
| 9 | Metrics logging + endpoint         | JSONL append + `GET /metrics/recent`                      |
| 10 | Benchmark enhancements            | Timestamped CSV+JSON + fallback columns                   |
| 11 | Manual evaluation harness         | `eval.yaml` + `benchmarks/evaluate.py` scoring            |
| 12 | Documentation + resume packaging  | This README + updated `docs/architecture.md`              |

Each phase was shipped as an atomic git commit against `main`.

## Design principles

- **Lazy singletons for expensive state.** Embedding model, Chroma collection,
  SW preference cache, and backend instances all follow the same three-part
  pattern (module-level cache + private producer + public accessor).
- **Data before code.** Preferences file, queries file, and eval file were
  authored before the scripts that consume them — curation drives
  implementation shape, not the other way around.
- **Observe before act.** Phase 6 made the router *visible* in metrics and
  responses before Phase 7 made it *change behavior*. Isolating observation
  from action keeps A/B rollouts clean.
- **Decide → prepare → execute.** The chat flow computes the route *before*
  building the prompt or calling the model, so future phases can specialize
  prompt format or backend selection per route without reshuffling the
  pipeline.
- **Config as data, not code.** Backend routing lives in a single dict
  (`BACKEND_CONFIG`) that maps labels to backend specs + fallback chains.
  Adding a new backend is one entry.

## Known limitations

- **Single-worker server.** Router config lives in process-local globals.
  Multi-worker deployment would need a shared config store (Redis, file).
- **Eval set is domain-specific.** The current `eval.yaml` evaluates against
  the Harry Potter / Ollivander sample document. For a new corpus, author a
  matching eval YAML; the scoring script is reusable.
- **No HTTP file upload yet.** `/documents/ingest` takes a server-side path.
  Direct uploads via multipart form data are a future enhancement.
- **CLOUD route is an alias.** Currently points at `llama3.2:3b` to prove the
  routing wiring. A real cloud backend (OpenAI, Anthropic) is a separate
  adapter class away.

## Future work (deferred)

- **OpenAI-compatible endpoint** (`POST /v1/chat/completions`) so AnythingLLM
  or other clients can use EdgeClaw as a generic provider.
- **File upload endpoint** with multipart `UploadFile`, path sanitization,
  size limits, and `.pdf` / `.docx` support.
- **Qualcomm NPU backend** (`QNNBackend`) using `qai-hub` compiled models for
  Snapdragon hardware — the dispatcher is designed to accommodate this with
  one new backend class.
- **N-way SW routing** — the paper notes N-way as future direction;
  implementable via two thresholds (`α_low`, `α_high`) over `P(strong_wins)`.
- **Arena preference ETL** — build a larger preference set by filtering the
  `lmarena-ai/arena-human-preference-55k` dataset.
- **Router config persistence** — survive restarts via a small on-disk JSON.
- **Advanced automatic eval** — LLM-as-judge, semantic similarity, grounding
  scores for domain-agnostic quality signals.

See `/local/mnt/workspace/notes/scratchpad/task/edgeclaw/active/todo-edgeclaw-python-srs-roadmap/todo.md`
(internal) for the full deferred-experiments backlog.

## References

- Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E.,
  Kadous, M. W., & Stoica, I. (2025).
  *RouteLLM: Learning to Route LLMs with Preference Data*. ICLR 2025.
  arXiv:2406.18665.
- Chatbot Arena (LMSYS) — `lmarena-ai/arena-human-preference-55k`.
- Chroma — https://www.trychroma.com/
- Ollama — https://ollama.com/
- Sentence-Transformers — https://www.sbert.net/
