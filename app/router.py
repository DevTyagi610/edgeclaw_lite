import logging
from app.rules import classify
from app.routers.similarity_router import route as similarity_route

logger = logging.getLogger("edgeclaw")

ROUTER_MODE: str = "rule"
ROUTER_ALPHA: float = 0.5

def route(query: str, context_size: int = 0) -> dict :

    if ROUTER_MODE == "similarity" :
        route_dict = similarity_route(query, ROUTER_ALPHA, context_size)
        return route_dict
    
    qlen = len(query.strip())
    if ROUTER_MODE != "rule" :
        logger.warning("Invalid routing method chosen, falling back to 'rule' method") 

    #fallback and ROUTER_MODE = rule
    route_dict = classify(query)    
        
    return {**route_dict, "query_length" : qlen, "context_size" : context_size}
    