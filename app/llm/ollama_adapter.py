import logging
from typing import List, Dict
from ollama import Client
from app.llm.base import LLMProvider, LLMResponse
from app.llm.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

class OllamaAdapter(LLMProvider):
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.client = Client(host=self.base_url)
        logger.info(f"Ollama adapter initialized with base_url: {self.base_url}, model: {self.model}")

    @property
    def name(self) -> str:
        return "Ollama"

    def map_exception(self, e: Exception) -> str:
        error_text = str(e)
        if "connection" in error_text.lower() or "refused" in error_text.lower() or "connect" in error_text.lower():
            return f"Ollama request failed: server unavailable or connection refused at {self.base_url}."
        elif "timeout" in error_text.lower():
            return "Ollama request failed: connection timeout."
        elif "not found" in error_text.lower() or "404" in error_text.lower():
            return f"Ollama request failed: model '{self.model}' not found on server. Please pull it first."
        return f"Ollama request failed: {error_text}"

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        logger.info(f"Sending completions request to Ollama using model: {self.model}")
        try:
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                if role == "assistant":
                    role = "assistant"
                formatted_messages.append({
                    "role": role,
                    "content": msg.get("content", "")
                })

            response = self.client.chat(
                model=self.model,
                messages=formatted_messages,
                format="json",
                options={
                    "temperature": temperature
                }
            )

            # Safely parse message response content
            # response can behave as a dict or Response object
            message_obj = response.get("message", {}) if hasattr(response, "get") else getattr(response, "message", {})
            content = ""
            if hasattr(message_obj, "get"):
                content = message_obj.get("content", "")
            else:
                content = getattr(message_obj, "content", "")

            # Parse token usage metrics dynamically
            prompt_tokens = 0
            completion_tokens = 0
            
            if hasattr(response, "get"):
                prompt_tokens = response.get("prompt_eval_count", 0) or 0
                completion_tokens = response.get("eval_count", 0) or 0
            else:
                prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
                completion_tokens = getattr(response, "eval_count", 0) or 0

            token_usage = {
                "INPUT": prompt_tokens,
                "OUTPUT": completion_tokens,
                "TOTAL": prompt_tokens + completion_tokens
            }

            logger.info("===== OLLAMA RAW CONTENT START =====")
            logger.info(content)
            logger.info("===== OLLAMA RAW CONTENT END =====")
                        
            return LLMResponse(
                content=content,
                token_usage=token_usage,
                provider_name="Ollama",
                model_name=self.model
            )
            
        except Exception as e:
            logger.exception(f"Ollama completions generation failed: {str(e)}")
            raise
