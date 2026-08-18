import logging
from sentence_transformers import SentenceTransformer
from typing import List
logger = logging.getLogger("edgeclaw")

model = None

#When called for 1st time,load the model else call the already loaded model. returns model
def get_model():
    global model
    if model is None :
        model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Loading the model: %s", model)
    return model

# Convert text chunks to embeddings using the model. returns embeddings of chunks -> List
def embed_texts(chunks: List[str]) -> List[List[float]]:
    #Empty text case :
    if len(chunks) == 0:
        return []
    else :
        logger.info("Converting chunks: %d to embeddings ", len(chunks))
        embedding_model = get_model()
        embeddings = embedding_model.encode(chunks)
    
    return embeddings.tolist()

# Conevert the text query to embeddings using the model. returns embeddings of chunks -> List
def embed_query(query: str) -> List[float]:
    #Empty text case :
    if len(query) == 0:
        return []
    else :
        logger.info("Converting user query: %d to embeddings ", len(query))
        embedding_model = get_model()
        embeddings = embedding_model.encode(query)
    
    return embeddings.tolist()
