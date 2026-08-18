from app import embeddings, vector_store
from typing import List

# For a user query embed the query and then return the list of top_k chunks 
def retrieve(query : str , top_k=3) -> List[dict]:
    #edge case -> empty query 
    if len(query) == 0 or not query.strip():
        return []
    embedding = embeddings.embed_query(query)  # returns embedded list for the query
    results = vector_store.query(embedding, top_k)  # returns the list of top_k chunks 

    return results
