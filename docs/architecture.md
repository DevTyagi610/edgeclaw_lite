# EdgeClaw Lite Architecture

This document explains the current EdgeClaw Lite system flow in beginner-friendly terms.

## What EdgeClaw Lite does right now

EdgeClaw Lite is a local RAG API service.

RAG means:

1. Take a user question.
2. Search local documents for useful context.
3. Build a prompt using that context.
4. Ask the local LLM to answer using the context.
5. Return the answer plus the source chunks used.

Current stack:

- API server: FastAPI
- Local LLM backend: Ollama with `llama3.2:3b`
- Embedding model: SentenceTransformer `all-MiniLM-L6-v2`
- Vector store: ChromaDB stored under `data/vector_store`
- Raw sample document: `data/raw/sample.md`
- Saved chunk JSON files: `data/chunks/<doc_id>.json`

## Runtime flow

### 1. Server startup

When the server starts with uvicorn:

1. FastAPI creates the web app.
2. `OllamaBackend` is created once.
3. The backend points to Ollama at `http://127.0.0.1:11434`.
4. The embedding model is not loaded immediately. It loads only when the first ingest or chat request needs embeddings.
5. The Chroma vector store is also opened only when it is first needed.

This is called lazy loading: expensive objects are created only when required.

### 2. Health check flow

Endpoint:

```text
GET /health
```

Purpose:

Check whether the API is alive and whether Ollama is reachable.

Flow:

```text
GET /health
  -> health()
  -> backend.is_available()
  -> Ollama client list request
  -> return backend_available true/false
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

Read a local document, split it into chunks, embed those chunks, and store them in the vector database.

Example request:

```json
{
  "path": "data/raw/sample.md"
}
```

High-level flow:

```text
file path
  -> read file text
  -> split text into chunks
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
  "doc_id": "some-id",
  "source": "sample.md",
  "num_chunks": 4,
  "saved_to": "data/chunks/some-id.json",
  "num_embedded": 4
}
```

Important meaning:

- `num_chunks` = how many text pieces were created.
- `num_embedded` = how many chunks were converted to vectors and stored.
- For a clean successful ingest, these two numbers should match.

### 4. Chat / RAG flow

Endpoint:

```text
POST /chat
```

Purpose:

Answer a user question using retrieved document context.

Example request:

```json
{
  "query": "What wand did Harry finally get?"
}
```

High-level flow:

```text
user question
  -> check Ollama is available
  -> embed the question
  -> search ChromaDB for top matching chunks
  -> build a RAG prompt from those chunks
  -> send prompt to Ollama
  -> return answer plus sources
```

Detailed call flow:

```text
main.chat(request)
  -> backend.is_available()
       -> Ollama client list request
  -> retriever.retrieve(request.query, top_k=3)
       -> embeddings.embed_query(query)
            -> embeddings.get_model()
            -> SentenceTransformer.encode(query)
       -> vector_store.query(query_embedding, top_k)
            -> vector_store.get_collection()
            -> Chroma collection.query(...)
            -> return chunks with text, source, doc_id, chunk_id, distance
  -> prompt_builder.build_rag_prompt(query, chunks)
       -> build system instruction
       -> add retrieved context
       -> add original question
  -> backend.generate(prompt)
       -> Ollama client chat request
       -> return BackendResponse
  -> build sources list from retrieved chunks
  -> return ChatResponse
```

Expected successful response shape:

```json
{
  "answer": "...",
  "query_length": 33,
  "backend": "ollama",
  "model": "llama3.2:3b",
  "latency_ms": 1234.5,
  "sources": [
    {
      "source": "sample.md",
      "chunk_id": 2,
      "distance": 0.75
    }
  ],
  "num_context_chunks": 3
}
```

Meaning of important fields:

- `answer`: the final LLM answer.
- `sources`: the retrieved chunks that were provided to the model.
- `source`: which document the chunk came from.
- `chunk_id`: which chunk number inside the document.
- `distance`: similarity distance from the query. Lower is generally closer.
- `num_context_chunks`: how many chunks were used in the prompt.
- `latency_ms`: how long the LLM generation took.

## Function responsibilities

### `app/main.py`

Owns the FastAPI endpoints.

Responsibilities:

- Define request and response models.
- Expose `/health`, `/documents/ingest`, `/documents/chunks`, and `/chat`.
- Orchestrate the full RAG flow for `/chat`.
- Convert internal results into API responses.
- Convert errors into HTTP status codes.

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
- Define the methods every backend should provide: generate and availability check.

### `app/ingestion.py`

Owns document ingestion.

Responsibilities:

- Validate file path and file type.
- Read `.txt` and `.md` files.
- Create chunk records.
- Save chunk JSON files for inspection.
- Send chunks for embedding.
- Store vectors in ChromaDB.

### `app/chunking.py`

Owns text splitting.

Responsibilities:

- Split long text into overlapping chunks.
- Keep chunks around 800 characters by default.
- Use 100 characters of overlap by default.

### `app/embeddings.py`

Owns embedding model loading and embedding generation.

Responsibilities:

- Load the SentenceTransformer model once.
- Convert document chunks into vectors.
- Convert user queries into vectors.

### `app/vector_store.py`

Owns ChromaDB access.

Responsibilities:

- Open or create the persistent ChromaDB collection.
- Add embedded document chunks.
- Query the vector store for similar chunks.
- Return chunks with text, metadata, and distance.

### `app/retriever.py`

Owns retrieval orchestration.

Responsibilities:

- Convert the user query into an embedding.
- Ask the vector store for the top matching chunks.
- Return the retrieved chunks to the chat endpoint.

### `app/prompt_builder.py`

Owns prompt construction.

Responsibilities:

- Build a prompt that tells the model to answer using only retrieved context.
- Include the retrieved chunks.
- Include the original user question.
- Handle the case where no relevant chunks are found.

## Important current behavior

### ChromaDB persists across server restarts

The vector store lives on disk at:

```text
data/vector_store
```

This means stored chunks remain even if you stop and restart uvicorn.

### `/documents/chunks` is only in-memory

The endpoint `/documents/chunks` reads from the `_INGESTED` list inside `ingestion.py`.

That list is in memory only.

So after a server restart:

- `/documents/chunks` may look empty.
- `/chat` can still retrieve from ChromaDB.

This is expected.

### Re-ingesting creates duplicates right now

Current ingestion generates a new random `doc_id` every time.

So if you ingest the same file many times, ChromaDB stores many copies of the same chunks.

This can hurt retrieval quality because duplicate chunks can fill all top results.

Current manual reset during learning:

```text
stop uvicorn
remove data/vector_store and data/chunks
restart uvicorn
ingest the document once
```

Future improvement:

Make ingestion idempotent. That means ingesting the same file again should replace or update the old chunks instead of adding duplicates.

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

If the answer says it does not know, likely causes are:

1. The document was not ingested.
2. The vector store contains duplicate stale data.
3. The answer chunk was not retrieved in the top context chunks.

## Mental model

Think of the system as four boxes:

```text
Documents -> Embeddings -> Vector Store -> LLM Answer
```

Or more fully:

```text
Ingest time:
Document -> chunks -> chunk embeddings -> ChromaDB

Question time:
Question -> query embedding -> retrieve chunks from ChromaDB -> prompt -> Ollama -> answer
```
