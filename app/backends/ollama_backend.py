import time
import logging
import ollama
from app.backends.base import BaseBackend, BackendResponse

logger = logging.getLogger("edgeclaw")

class OllamaBackend(BaseBackend):
    name = "ollama"
    def __init__(    # -> Constructor, i.e. def __init__ will run automatically whenever object of this class is created
        self,        # -> Refers to the current object.
        model: str = "llama3.2:3b",  # -> default model to be used if no model is provided
        host: str = "http://127.0.0.1:11434",   # -> addr of Ollama server
        timeout: int = 120,   # -> wait for 120s for response. If no resp -> Timeout Error
    ):
        self.model = model  # -> Store model in object
        self.host = host
        # Create resulable conn to Ollama.after this selt.client can be used to send reqs
        self.client = ollama.Client(host=host, timeout=timeout)  

    # Check if the model is available or not
    def is_available(self) -> bool:
        """Return True if Ollama answers, False if it's unreachable."""
        try:
            self.client.list()
            return True
        except Exception as e:
            logger.warning("Ollama not available: %s", e)
            return False
        
    # Accept a text prompt and return a BackendResponse
    def generate(self, prompt: str) -> BackendResponse:
        """Send the prompt to the model and return a BackendResponse."""
        start = time.perf_counter()
        try:
            resp = self.client.chat(   # -> Calling Ollama's chat API
                model=self.model,      # -> Use the selected model
                stream=False,          # -> return entire resp at once, not token by token
                messages=[{"role": "user", "content": prompt}],  # -> Send the prompt as a user message
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return BackendResponse(   # -> build response object 
                text=resp.message.content,  # -> Extract generated answer.
                backend=self.name,      # -> stores " Ollama"
                model=self.model,       # -> stores model name
                latency_ms=round(latency_ms, 1),
            )
        # If Exception, return response obj with error details
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("Ollama generation failed: %s", e)
            return BackendResponse(   
                text="",
                backend=self.name,
                model=self.model,
                latency_ms=round(latency_ms, 1),
                error=str(e),
            )