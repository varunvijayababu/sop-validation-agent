import logging
from typing import List, Dict
from groq import Groq
from app.llm.base import LLMProvider, LLMResponse
from app.llm.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

class GroqAdapter(LLMProvider):
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL
        logger.info(f"Groq adapter initialized with model: {self.model}")

    @property
    def name(self) -> str:
        return "Groq"

    def map_exception(self, e: Exception) -> str:
        import groq
        error_text = str(e)
        if isinstance(e, groq.RateLimitError) or "rate_limit_exceeded" in error_text or "Rate limit reached" in error_text or "429" in error_text:
            return "Groq request failed: rate limit exceeded. Please retry later."
        elif isinstance(e, groq.AuthenticationError):
            return "Groq request failed: authentication error. Please verify your API key."
        elif isinstance(e, groq.BadRequestError):
            if "Request too large" in error_text or "413" in error_text:
                return "Groq request failed: validation request exceeds model token limits. Reduce SOP size or retrieved context."
            return f"Groq request failed: bad request. {error_text}"
        elif isinstance(e, groq.APIConnectionError):
            return "Groq request failed: connection error or timeout. Please check your network."
        return f"Groq request failed: {error_text}"

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        logger.info(f"Sending completions request to Groq using model: {self.model}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content
            
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if hasattr(response, "usage") and response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)
                total_tokens = getattr(response.usage, "total_tokens", 0)

            token_usage = {
                "INPUT": prompt_tokens,
                "OUTPUT": completion_tokens,
                "TOTAL": total_tokens or (prompt_tokens + completion_tokens)
            }
            
            return LLMResponse(
                content=content,
                token_usage=token_usage,
                provider_name="Groq",
                model_name=self.model
            )
            
        except Exception as e:
            logger.exception(f"Groq completions generation failed: {str(e)}")
            raise
