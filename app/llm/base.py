from abc import ABC, abstractmethod
from typing import Dict, Any, List

class LLMResponse:
    def __init__(self, content: str, token_usage: Dict[str, int] = None, provider_name: str = "", model_name: str = ""):
        self.content = content
        self.token_usage = token_usage or {
            "INPUT": 0,
            "OUTPUT": 0,
            "TOTAL": 0
        }
        self.provider_name = provider_name
        self.model_name = model_name

class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the provider name (e.g. 'Groq', 'OpenAI').
        """
        pass

    @abstractmethod
    def map_exception(self, e: Exception) -> str:
        """
        Maps provider-specific exceptions to human-friendly strings.
        """
        pass

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        """
        Generates a completion response from the LLM provider.
        """
        pass
