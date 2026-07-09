import logging
from typing import List, Dict
from openai import OpenAI
from app.llm.base import LLMProvider, LLMResponse
from app.llm.config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

class OpenAIAdapter(LLMProvider):
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        logger.info(f"OpenAI adapter initialized with model: {self.model}")

    @property
    def name(self) -> str:
        return "OpenAI"

    def map_exception(self, e: Exception) -> str:
        import openai
        error_text = str(e)
        if isinstance(e, openai.RateLimitError) or "rate_limit_exceeded" in error_text or "429" in error_text:
            if "insufficient_quota" in error_text or "billing" in error_text:
                return "OpenAI request failed: insufficient quota. Please check your API billing or usage limits."
            return "OpenAI request failed: rate limit exceeded. Please retry later."
        elif isinstance(e, openai.AuthenticationError) or "invalid_api_key" in error_text:
            return "OpenAI request failed: authentication error. Please verify your API key."
        elif isinstance(e, openai.NotFoundError) or "model_not_found" in error_text:
            return f"OpenAI request failed: model not found. {error_text}"
        elif isinstance(e, openai.BadRequestError):
            if "Request too large" in error_text or "413" in error_text:
                return "OpenAI request failed: validation request exceeds model token limits. Reduce SOP size or retrieved context."
            return f"OpenAI request failed: bad request. {error_text}"
        elif isinstance(e, openai.APIConnectionError):
            return "OpenAI request failed: connection error or timeout. Please check your network."
        return f"OpenAI request failed: {error_text}"

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        logger.info(f"Sending completions request to OpenAI using model: {self.model}")
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
                provider_name="OpenAI",
                model_name=self.model
            )
            
        except Exception as e:
            logger.exception(f"OpenAI completions generation failed: {str(e)}")
            raise
