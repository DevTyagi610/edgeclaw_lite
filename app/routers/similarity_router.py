import logging
import os
import json
from app import embeddings
import numpy as np

logger = logging.getLogger("edgeclaw")

# Global constants
PREFERENCES_PATH = "benchmarks/preferences.jsonl"
GAMMA = 10.0
_LABEL_TO_NUM = {"strong_wins": 1.0, "weak_wins": 0.0, "tie": 0.5}
_cache = None

def _load_preferences() -> tuple :
    records = []  #list[dict]
    queries = []  # list[str]
    labels = []  # list[float]
    try: 
        with open(PREFERENCES_PATH, "r", encoding="utf-8") as file :
            for line in file : 
                line = line.strip()
                if line : # Skip empty lines
                    records.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error loading jsonl file: {e}")
        return (None, None)

    index = 0
    for record in records :
        index = index + 1
        if record["label"] not in _LABEL_TO_NUM :
            logger.info(f"Skipping record {index} , reason: label is wrong")
            continue
        else :
            queries.append(record["query"])
            labels.append(_LABEL_TO_NUM[record["label"]])

    N = len(queries)   # Number of preferences
    if  N==0 :
        logger.error(f"0 valid rows in {PREFERENCES_PATH}")
        return (None, None)
    embeddings_lst = embeddings.embed_texts(queries)
    # D = len(embeddings_lst[0])  # dimension of 1 embedding
    pref_matrix = np.array(embeddings_lst)

    pref_matrix = pref_matrix/np.linalg.norm(pref_matrix, axis=1, keepdims= True)
    label_arr = np.array(labels)
    logger.info(f"Loaded {N} preferences from {PREFERENCES_PATH}")

    return (pref_matrix , label_arr)

# A wrapper func
def _get_preferences_cache() -> tuple :
    global _cache
    if _cache is None :
        _cache = _load_preferences()

    return _cache


def predict_strong_win_prob(query: str) -> float | None :
    pref_matrix , label_arr = _get_preferences_cache()

    if pref_matrix is None :
        return None

    query_vec = np.array(embeddings.embed_query(query))
    if len(query_vec) == 0 :
        return None
    query_vec = query_vec/np.linalg.norm(query_vec)
    # Shape of co_sim = (N,) <- 1D array of size N (num of preferences)
    co_sim = np.matmul(pref_matrix, query_vec)   

    weights = GAMMA**(1+co_sim)  # Shape = (N,)

    # Weighted mean of labels
    p_strong = float((weights*label_arr).sum() / weights.sum())

    return p_strong

def route(query : str, alpha : float = 0.5, context_size: int = 0 ) -> dict :
    p_strong = predict_strong_win_prob(query)
    query_len = len(query.strip())

    if p_strong is None :
        route_chosen = "LOCAL_FALLBACK"
        reason = "similarity router unavailable — using safe default"
        
    elif p_strong < alpha :
        route_chosen = "LOCAL_FAST"
        reason = f"P_strong {p_strong:.3f} is less than threshold:{alpha}"
    else : 
        route_chosen = "CLOUD"
        reason = f"P_strong {p_strong:.3f} is greater than or equal to threshold:{alpha}"

    return {"route" : route_chosen, "reason" : reason, "query_length": query_len,
            "context_size" : context_size, "p_strong_wins" : p_strong, "alpha" : alpha}