import logging
from app.llm.config import LLM_PROVIDER
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Registry of provider classes
_PROVIDER_REGISTRY = {}

def register_provider(name: str, provider_class):
    """Register a provider adapter class case-insensitively and stripped."""
    _PROVIDER_REGISTRY[name.lower().strip()] = provider_class

# Register default provider adapters
from app.llm.groq_adapter import GroqAdapter
from app.llm.openai_adapter import OpenAIAdapter
from app.llm.gemini_adapter import GeminiAdapter
from app.llm.ollama_adapter import OllamaAdapter

register_provider("groq", GroqAdapter)
register_provider("openai", OpenAIAdapter)
register_provider("gemini", GeminiAdapter)
register_provider("ollama", OllamaAdapter)

def get_llm_provider() -> LLMProvider:
    logger.info(f"Resolving LLM provider for selection: '{LLM_PROVIDER}'")
    key = LLM_PROVIDER.lower().strip()
    if key not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported LLM provider: '{LLM_PROVIDER}'")
    provider_cls = _PROVIDER_REGISTRY[key]
    return provider_cls()
