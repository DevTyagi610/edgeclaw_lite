import json
import os
import logging

logger = logging.getLogger("edgeclaw")

METRICS = os.path.join("data", "metrics")

def log_chat_event(event: dict) -> None :
    os.makedirs(METRICS, exist_ok=True)

    out_path = os.path.join(METRICS, "chat.jsonl")

    try : 
        with open(out_path, "a", encoding="utf-8" ) as f :
            json.dump(event , f, ensure_ascii=False)
            f.write("\n")
    except Exception as e :
        logger.warning(f"Could not write chat metrics to disk, Exception : {e}")

# Reading the latest 20 metrics json 
def read_recent(limit: int = 20) -> list[dict] :
    # If limit is negative or zero
    if limit <= 0 :
        limit = 20

    metric_lst = []

    path = os.path.join(METRICS ,"chat.jsonl")

    # Path doesnt exist ( no req has been made yet, return empty list)
    if not os.path.exists(path) :
        return []
    
    with open(path, "r", encoding="utf-8") as file :
        for line_number, line in enumerate(file, start=1) :
            if not line.strip():
                continue
            try :
                data = json.loads(line)
                metric_lst.append(data)
            except json.JSONDecodeError as e: 
                logger.warning(f"Error parsing line at line number: {line_number} , error : {e}")

    limited_lst = metric_lst[-limit:]
    limited_lst.reverse()

    return limited_lst
