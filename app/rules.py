# Created a rule to classify the query to its route with reason
def classify(query: str) -> dict :
    qlen = len(query.strip())
    if qlen == 0 :
        return {"route" : "LOCAL_FAST", "reason" :  "Empty query"}
    q_lower = query.lower()

    keyword_lst = ["summarize", "analyze", "compare"]

    # matched will either output list with keywords if they are present in query or None otherwise
    matched = next((k for k in keyword_lst if k in q_lower), None)

    if matched :
        route = "CLOUD"
        reason = f"Keyword : {matched} mentioned in query"
    elif qlen <= 80 : 
        route = "LOCAL_FAST"
        reason = "Short query <=80 length, can be done by local Fast model"
    else :
        route = "LOCAL_FALLBACK"
        reason = "Default routing"

    return {"route" : route , "reason" : reason}





