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
