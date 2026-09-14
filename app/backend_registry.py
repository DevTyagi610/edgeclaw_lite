import logging
from app.backends.ollama_backend import OllamaBackend
from app.backends.base import BaseBackend

logger = logging.getLogger("edgeclaw")

BACKEND_CONFIG = {
    "LOCAL_FAST" : {"type": "ollama", "model" : "llama3.2:1b"},
    "LOCAL_FALLBACK" : {"type": "ollama", "model" : "llama3.2:3b"},
    "CLOUD" : {"type": "ollama", "model" : "llama3.2:3b"}
}

DEFAULT_LABEL = "LOCAL_FALLBACK"
_cache = {}   #Will hold {route_label: BaseBackend} mappings


def get_backend(route_label: str) -> BaseBackend:
    if route_label not in BACKEND_CONFIG:
        logger.warning(f"Invalid label : {route_label}, falling back to Default Label")
        route_label = DEFAULT_LABEL   

    if route_label in _cache :
        return _cache[route_label]

    # route_label NOT in _cache 
    cfg = BACKEND_CONFIG[route_label]   # cfg is name of backend_config
        
    if cfg["type"] != "ollama" : 
        raise ValueError(f"Unknown Backend type : {cfg['type']}")

    backend = OllamaBackend(model = cfg["model"])
    _cache[route_label] = backend


    return _cache[route_label]