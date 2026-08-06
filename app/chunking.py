from typing import List

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split `text` into pieces of about `chunk_size` characters,
    where each piece shares `overlap` characters with the previous one.
    Returns a list of chunk strings.
    """
    # --- guard against silly settings ---
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    # --- clean up and handle the empty case ---
    text = text.strip()
    if not text:
        return []
    
    chunks: List[str] = []
    start = 0
    n = len(text)

    # --- walk through the text in steps ---
    while start < n:
        end = start + chunk_size          # where this chunk ends
        chunks.append(text[start:end])    # grab the slice
        if end >= n:                      # reached the end -> stop
            break
        start = end - overlap             # step forward, but back up by `overlap`

    return chunks