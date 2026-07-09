import logging
from typing import List, Dict
from google import genai
from google.genai import types
from app.llm.base import LLMProvider, LLMResponse
from app.llm.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

class GeminiAdapter(LLMProvider):
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL
        logger.info(f"Gemini adapter initialized with model: {self.model}")

    @property
    def name(self) -> str:
        return "Gemini"

    def map_exception(self, e: Exception) -> str:
        from google.genai import errors
        error_text = str(e)
        if isinstance(e, errors.APIError):
            status_code = getattr(e, "code", None)
            if status_code == 429 or "rate" in error_text.lower() or "quota" in error_text.lower():
                return "Gemini request failed: rate limit or quota exceeded. Please check your billing or retry later."
            elif status_code == 401 or status_code == 403 or "key" in error_text.lower():
                return "Gemini request failed: authentication error. Please verify your API key."
            elif status_code == 404 or "model" in error_text.lower():
                return f"Gemini request failed: model not found. {error_text}"
            return f"Gemini request failed: API error ({status_code}). {error_text}"
        return f"Gemini request failed: {error_text}"

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        logger.info(f"Sending completions request to Gemini using model: {self.model}")
        try:
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                if role == "assistant":
                    role = "model"
                content = msg.get("content", "")
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=content)]
                    )
                )

            config = types.GenerateContentConfig(
                temperature=temperature
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            content = response.text or ""
            
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
                total_tokens = getattr(response.usage_metadata, "total_token_count", 0)

            token_usage = {
                "INPUT": prompt_tokens,
                "OUTPUT": completion_tokens,
                "TOTAL": total_tokens or (prompt_tokens + completion_tokens)
            }
            
            return LLMResponse(
                content=content,
                token_usage=token_usage,
                provider_name="Gemini",
                model_name=self.model
            )
            
        except Exception as e:
            logger.exception(f"Gemini completions generation failed: {str(e)}")
            raise
