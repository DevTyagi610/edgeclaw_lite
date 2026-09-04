from app.rules import classify

def route(query: str, context_size: int = 0) -> dict :
    route_dict = classify(query)
    qlen = len(query.strip())
    return {**route_dict, "query_length" : qlen, "context_size" : context_size}
