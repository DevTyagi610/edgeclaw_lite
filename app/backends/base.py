from dataclasses import dataclass
from typing import Optional
@dataclass
class BackendResponse:
    """A standard container for whatever a backend returns."""
    text: str            # the model's answer
    backend: str         # which backend produced it, e.g. "ollama"
    model: str           # which model, e.g. "llama3.2:3b"
    latency_ms: float    # how long generation took, in milliseconds
    error: Optional[str] = None  # filled in only if something went wrong
class BaseBackend:
    """The 'contract'. Every real backend must provide these two methods."""
    name = "base"
    def generate(self, prompt: str) -> BackendResponse:
        raise NotImplementedError
    def is_available(self) -> bool:
        raise NotImplementedError