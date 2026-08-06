import os
import json
import uuid
import logging
from typing import List, Dict
from app.chunking import chunk_text


logger = logging.getLogger("edgeclaw")

# Which file types we accept for now.
ALLOWED_EXTENSIONS = {".txt", ".md"}

# Where to save chunks on disk so you can open and inspect them.
CHUNKS_DIR = os.path.join("data", "chunks")


# A simple in-memory store: every chunk we've ingested this session.
# Each item looks like: {"doc_id", "source", "chunk_id", "text"}

# underscore means The var is intended for internal use in this module
_INGESTED: List[Dict] = []  # -> Means _INGESTED is a list which is expected to stores dicts

def load_document(path: str) -> str:
    """Read a text file from disk and return its contents as a string."""
    _, ext = os.path.splitext(path)   # -> "_" means ignore the 1st val.
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

    
def ingest_file(path: str) -> Dict:
    """
    Full ingestion for one file:
    read -> chunk -> store in memory -> save a JSON copy to disk.
    Returns a small summary dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    text = load_document(path)
    pieces = chunk_text(text)

# uuid.uuid4() generates a random unique id, which we then typescast to str and slice 1st 8 chars
    doc_id = str(uuid.uuid4())[:8]     
    source = os.path.basename(path)    # just the filename, e.g. "sample.md"
    records = []

    #enumerate helps here to know which chunk is the 1st, 2nd, or 3rd 
    # by conveniently providing that index as i.
    for i, piece in enumerate(pieces):
        records.append({
            "doc_id": doc_id,
            "source": source,
            "chunk_id": i,
            "text": piece,
        })

    # Add to the in-memory store.
    _INGESTED.extend(records)

    # Also save a copy to disk so you can open it and see the chunks.
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    out_path = os.path.join(CHUNKS_DIR, f"{doc_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Ingested '%s' -> %d chunks (doc_id=%s)", source, len(records), doc_id)
    return {
        "doc_id": doc_id,
        "source": source,
        "num_chunks": len(records),
        "saved_to": out_path,
    }


def get_chunks() -> List[Dict]:
    """Return all chunks ingested so far this session."""
    return _INGESTED