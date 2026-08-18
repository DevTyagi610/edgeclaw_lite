import logging
from typing import List
import chromadb
from itertools import islice

logger = logging.getLogger("edgeclaw")

client = None

#Create a collection when called for 1st time, else use already made collection
def get_collection():
    global client
    if client is None: 
        logger.info("Creating a persistent client")
        client = chromadb.PersistentClient(path= "data/vector_store")        
    collection = client.get_or_create_collection("edgeclaw_chunks")
    logger.info("Using the collection: edgeclaw_chunks")

    return collection

# In the collection, add the chunks and their embeddings to the collection, return num of records added
def add_chunks(records: List[dict] ,embeddings: List[List[float]]) -> int:
    collection = get_collection()

    # Empty records edge case
    if len(records) == 0 :
        return 0
    elif len(records) != len(embeddings) :
        raise ValueError(f"Length of records {len(records)} does not match embeddings {len(embeddings)}")
    else:         
        ids, documents, metadatas, embedding = ([] for _ in range(4))
        for i in range(0, len(records)):
                ids.append(records[i]["doc_id"] + "_" + str(records[i]["chunk_id"]))
                documents.append(records[i]["text"])
                metadatas.append({"doc_id" :records[i]["doc_id"], "source": records[i]["source"], 
                              "chunk_id": records[i]["chunk_id"] })
                embedding.append(embeddings[i])     
    
    collection.add(ids=ids, documents=documents , metadatas=metadatas, embeddings=embedding)
    return len(records)


#Takes the embedded query and top_k , and returns the top_k chunks based on distance which is list of dict 
def query(query_embedding: List[float] , top_k=3) -> List[dict]:
    collection = get_collection()
    top_queries = collection.query(query_embeddings = [query_embedding],
                      n_results = top_k,
                      include = ["documents", "metadatas", "distances"])

    docs = top_queries["documents"][0]
    metas = top_queries["metadatas"][0]
    dists = top_queries["distances"][0]

    lst = []
    for text, meta, dist in zip(docs, metas, dists):
         lst.append({
              "text" : text,
              "source" : meta["source"],
              "doc_id" : meta["doc_id"],
              "chunk_id" : meta["chunk_id"],
              "distance" : dist
         })

    return lst


    


